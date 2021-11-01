# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 01:46:32 2021

@author: 14488
"""

"""Example Gym environment for the RRC 2021 Phase 2."""
import enum
import pathlib
import typing

import gym
import numpy as np

import robot_fingers
from torch.autograd import Variable
import rrc_example_package.trifinger_simulation.python.trifinger_simulation as trifinger_simulation
import rrc_example_package.trifinger_simulation.python.trifinger_simulation.tasks.rearrange_dice as task
from rrc_example_package.trifinger_simulation.python.trifinger_simulation import trifingerpro_limits
from rrc_example_package.trifinger_simulation.python.trifinger_simulation.camera import load_camera_parameters #, change_param_image_size
import rrc_example_package.trifinger_simulation.python.trifinger_simulation.visual_objects

from trifinger_cameras.utils import convert_image
from trifinger_object_tracking.py_lightblue_segmenter import segment_image

import cv2
import matplotlib.pyplot as plt
import os
import copy
import torch
import torchvision.transforms as T

Encoder = torch.load('./Encoder_140.pt')
# Decoder = torch.load('./Decoder_500.pt')
Dec = torch.load('./Dec_140.pt')
# Inc = torch.load('./Inc_500.pt')
latent_dim = 64
state_dim = 9  * 3
goal_dim = 6
obs_dim = latent_dim + state_dim + goal_dim
# obs_dim = 102

gpu_id = 0
device = torch.device("cuda", gpu_id)

CONFIG_DIR = pathlib.Path("/etc/trifingerpro")

SIM_CONFIG_DIR = pathlib.Path("src/rrc_example_package/camera_params")

SIM_CALIB_FILENAME_PATTERN = "camera{id}_cropped_and_downsampled.yml"


def preprocess_observation_(observation, bit_depth):
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


def _images_to_observation(images, bit_depth):
  # print(images)
  # images = torch.tensor(cv2.resize(images, (128,128), interpolation=cv2.INTER_LINEAR).transpose(0, 1, 2), dtype=torch.float32)  # Resize and put channel first
  images = torch.tensor(cv2.resize(images, (128,128), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  # images = cv2.resize(images, (128,128), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)  # Resize and put channel first
   # images = torch.tensor(cv2.resize(images, (64,64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  # preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
  # return images.unsqueeze(dim=0)  # Add batch dimension
  return images.unsqueeze(dim=0)

Trans =  T.Compose([T.ToTensor(),T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class ActionType(enum.Enum):
    """Different action types that can be used to control the robot."""

    #: Use pure torque commands.  The action is a list of torques (one per
    #: joint) in this case.
    TORQUE = enum.auto()
    #: Use joint position commands.  The action is a list of angular joint
    #: positions (one per joint) in this case.  Internally a PD controller is
    #: executed for each action to determine the torques that are applied to
    #: the robot.
    POSITION = enum.auto()
    #: Use both torque and position commands.  In this case the action is a
    #: dictionary with keys "torque" and "position" which contain the
    #: corresponding lists of values (see above).  The torques resulting from
    #: the position controller are added to the torques in the action before
    #: applying them to the robot.
    TORQUE_AND_POSITION = enum.auto()


class SimtoRealRearrangeDiceEnv(gym.GoalEnv):
    """Gym environment for rearranging dice with a TriFingerPro robot."""

    def __init__(
        self,
        provided_goal: typing.Optional[task.Goal] = None,
        action_type: ActionType = ActionType.POSITION,
        step_size: int = 2,
        sim: bool = True,
        visualization: bool = False,
        enable_cameras: bool = False,
        num_dice: int = 2,
        max_steps: int = 500,
        image_size=270,
        distance_threshold = 0.01,
        include_dice_velocity = True,
        include_dice_orient = True,
        flat_all = False,
        AE = False
    ):
        """Initialize.
        Args:
            goal: Goal pattern for the dice.  If ``None`` a new random goal is
                sampled upon reset.
            action_type: Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size:  Number of actual control steps to be performed in one
                call of step().
        """
        # Basic initialization
        # ====================

        if provided_goal is not None:
            task.validate_goal(provided_goal)
        self.provided_goal = provided_goal
        self.action_type = action_type
        self.sim = sim
        self.visualization = visualization
        self.enable_cameras = enable_cameras
        self.num_dice = num_dice
        self._max_episode_steps = max_steps
        task.EPISODE_LENGTH = max_steps * step_size
        self.image_size = image_size
        self.distance_threshold = distance_threshold
        self.include_dice_velocity = include_dice_velocity
        self.include_dice_orient = include_dice_orient
        self.flat_all = flat_all
        self.AE = AE
        
        if num_dice < 1 or num_dice > 25:
            raise ValueError("num_dice must be > 0 and < 26.")
        else:
            task.NUM_DICE = num_dice
            
        if step_size < 1:
            raise ValueError("step_size cannot be less than 1.")
        else:
            self.step_size = step_size

        # will be initialized in reset()
        self.platform = None
        
        # load camera parameters
        if sim:
            param_dir = SIM_CONFIG_DIR
        else:
            param_dir = CONFIG_DIR
        self.camera_params = load_camera_parameters(
            param_dir, "camera{id}_cropped_and_downsampled.yml"
        )
        # self.camera_params = change_param_image_size(self.camera_params, image_size=image_size)

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        mask_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 270, 270), dtype=np.uint8
        )

        if self.action_type == ActionType.TORQUE:
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")


        if self.flat_all:
            self.low_state = np.array(
            -255 * np.ones((obs_dim,),dtype=np.float32)
            )
            
            self.high_state = np.array(
            255 * np.ones((obs_dim,),dtype=np.float32)
            )
            
            self.observation_space = gym.spaces.Box(
                low=self.low_state,
                high=self.high_state,
                dtype=np.float32
            )
            
        else:
            # self.observation_space = gym.spaces.Dict(
            #     {
            #         "robot_observation": gym.spaces.Dict(
            #             {
            #                 "position": robot_position_space,
            #                 "velocity": robot_velocity_space,
            #                 "torque": robot_torque_space,
            #             } # TODO: add tip forces?
            #         ),
            #         "action": self.action_space,
            #         "desired_goal": mask_space,
            #         "achieved_goal": mask_space,
            #     }
            # )
            self.observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(-255.0, 255.0, shape=(latent_dim,), dtype='float32'),
            achieved_goal=gym.spaces.Box(-255.0, 255.0, shape=(latent_dim,), dtype='float32'),
            observation=gym.spaces.Box(-255.0, 255.0, shape=(state_dim,), dtype='float32'),
            ))

    def compute_reward(
        self,
        achieved_goal: typing.Sequence[np.ndarray],
        desired_goal: typing.Sequence[np.ndarray],
        info: dict,
    ) -> float:
        """Compute the reward for the given achieved and desired goal.
        Args:
            achieved_goal: Segmentation mask of the observed camera images.
            desired_goal: Segmentation mask of the goal positions.
            info: Unused.
        Returns:
            The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal. Note that the following should always hold true::
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'],
                    ob['desired_goal'],
                    info,
                )
        """
        if self.enable_cameras:
            return -task.evaluate_state(desired_goal, achieved_goal)
        else:
            return self.sparse_reward(desired_goal, achieved_goal)
        
    def sparse_reward(self, desired_goal, achieved_goal):
        """
        If a goal has been achieved (a dice is within threshold), its reward is 0. Else it is -1
        """
        # TODO: Check works properly!!!!!
        # TODO: atm 1 dice can achieve multiple goals. Undo?
        g_dim = 3
        r_shape = list(desired_goal.shape)
        r_shape[-1] = self.num_dice
        reward = np.zeros(tuple(r_shape))
        assert self.num_dice == int(desired_goal.shape[-1] / g_dim)
        assert desired_goal.shape == achieved_goal.shape
        for g in range(self.num_dice):
            g_idx = g * g_dim
            check_g = desired_goal[..., g_idx:g_idx+g_dim]
            for ag in range(self.num_dice):
                ag_idx = ag * g_dim
                check_ag = achieved_goal[..., ag_idx:ag_idx+g_dim]
                d = np.linalg.norm(check_ag - check_g, axis=-1)
                reward[..., g] += d < self.distance_threshold
        reward = reward <= 0
        # Mean ensures reward at each step is between 0 and -1
        reward = -np.mean(reward, axis=-1)
        return reward
                        
    def seed(self, seed=None):
        """Sets the seed for this env’s random number generator.
        .. note::
           Spaces need to be seeded separately.  E.g. if you want to sample
           actions directly from the action space using
           ``env.action_space.sample()`` you can set a seed there using
           ``env.action_space.seed()``.
        Returns:
            List of seeds used by this environment.  This environment only uses
            a single seed, so the list contains only one element.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        task.seed(seed)
        return [seed]

    def _create_observation(self, t, action):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)
        
        if self.enable_cameras:
            if self.sim:
                segmentation_masks = [
                    segment_image(cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR))
                    for c in camera_observation.cameras
                ]
            else:
                segmentation_masks = [
                    segment_image(convert_image(c.image))
                    for c in camera_observation.cameras
                ]
            observation = {
                "robot_observation": {
                    "position": robot_observation.position,
                    "velocity": robot_observation.velocity,
                    "torque": robot_observation.torque,
                    "tip_force": robot_observation.tip_force
                },
                "action": action,
                "desired_goal": self.goal_masks,
                "achieved_goal": segmentation_masks,
            }
        else:
            dice_pos, dice_orient = self.platform.get_dice_states()
            observation = {
                "robot_observation": {
                    "position": robot_observation.position,
                    "velocity": robot_observation.velocity,
                    "torque": robot_observation.torque,
                    "tip_force": robot_observation.tip_force
                },
                "dice_observation": {
                    "positions": dice_pos,
                    "orientations": dice_orient
                },
                "action": action,
                "desired_goal": self.goal, # TODO: remove redundant z dimension?? - but beware effects on reward and HER calcs
                "achieved_goal": dice_pos,
            }
        return observation
    
    def _flatten_obs(self, obs):
        # TODO: speedup operations?
        # TODO: include tipforces?, remove torque?, include previous action?
        state_obs = obs["robot_observation"]["position"]
        state_obs = np.concatenate((state_obs, obs['robot_observation']['velocity']))
        state_obs = np.concatenate((state_obs, obs['robot_observation']['torque']))
        # state_obs = np.concatenate((state_obs, obs['robot_observation']['tip_force']))
        # If using states rather than images
        if not self.enable_cameras:
            positions = np.array(obs["dice_observation"]['positions']).flatten()
            state_obs = np.concatenate((state_obs, positions))
            # WARNING: Not accounting for camera update delays!!!
            if self.include_dice_velocity:
                if self._prev_dice_pos is None:
                    # If start of episode, velocity is 0
                    self._prev_dice_pos = positions
                # Take velocity as diff in position
                pos_vel = positions - self._prev_dice_pos
                # pos_vel = self._prev_dice_pos
                state_obs = np.concatenate((state_obs, pos_vel))
                # Set new prev position
                self._prev_dice_pos = positions
            if self.include_dice_orient:
                orientations = np.array(obs["dice_observation"]['orientations']).flatten()
                state_obs = np.concatenate((state_obs, orientations))
                if self.include_dice_velocity:
                    if self._prev_dice_orient is None:
                        # If start of episode, velocity is 0
                        self._prev_dice_orient = orientations
                    # Take velocity as diff in position
                    orient_vel = orientations - self._prev_dice_orient
                    state_obs = np.concatenate((state_obs, orient_vel))
                    # Set new prev orientation
                    self._prev_dice_orient = orientations
        else:
            # raise NotImplementedError()
        
            if self.AE:
                des = self._AE(obs["desired_goal"])
                ach = self._AE(obs["achieved_goal"])
            else:
                des = np.array(obs["desired_goal"]).flatten()
                ach = np.array(obs["achieved_goal"]).flatten()
                
            flat_obs = {
                "observation": state_obs,
                "desired_goal": des,
                "achieved_goal": ach,
                }
            
        return flat_obs
    
    def _AE(self,mask):
        mask = np.array(mask)   # TODO: the stucture here is not efficient, need to be improved.
        mask = mask.transpose(1,2,0)
        mask = _images_to_observation(mask, 5)
        # mask = mask.numpy()
        # mask = Trans(mask).to(device)
        # k1 = copy(mask)
        # k1 = k1.unsqueeze(dim=0)
        # mask = mask.unsqueeze(dim=0).to(device)
        # k1 = copy(mask)
        mask = mask.data
        mask = Variable(mask).type('torch.cuda.FloatTensor').to(device)
        mask = Encoder(mask)
        mask = Dec(mask)
        # mask = Inc(mask)
        # mask = Decoder(mask)
        mask = mask.cpu().detach().numpy()
        
        return mask
    
    def _build_obs(self,obs):
        
        state_obs = obs["robot_observation"]["position"]
        # state_obs = state_obs.repeat(3)
        state_obs = np.concatenate((state_obs, obs['robot_observation']['velocity']))
        state_obs = np.concatenate((state_obs, obs['robot_observation']['torque']))
        
        if self.enable_cameras:
            des = copy.copy(np.array(self.goal))
            des = des.reshape(6,)
            # des = des.repeat(4)
            
            ach = self._AE(obs["achieved_goal"])
            ach = ach.reshape(latent_dim,)
            
        obs = np.concatenate((state_obs,des,ach))
        # obs = np.concatenate((state_obs,des))
        # print(obs)
        
        return obs
    
    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def step(self, action, initial=False):
        """Run one timestep of the environment's dynamics.
        Important: ``reset()`` needs to be called before doing the first step.
        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).
        Returns:
            tuple:
            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            print('Action space: {}'.format(self.action_space))
            print('Input action: {}'.format(action))
            raise ValueError(
                "Given action is not contained in the action space."
            )
        
        num_steps = self.step_size

        # ensure episode length is not exceeded due to step_size
        step_count_after = self.info["time_index"] + num_steps
        if step_count_after > task.EPISODE_LENGTH:
            excess = step_count_after - task.EPISODE_LENGTH
            num_steps = max(1, num_steps - excess)
            
        # TODO: fix actions if domain randomizing

        reward = 0.0
        robot_action = self._gym_action_to_robot_action(action)
        for _ in range(num_steps):
            # send action to robot
            t = self.platform.append_desired_action(robot_action)
            # make sure to not exceed the episode length
            if initial or t >= task.EPISODE_LENGTH - 1:
                break
            
        self.info["time_index"] = t    
        observation = self._create_observation(t, action)
        
        reward += self.compute_reward(
            observation["achieved_goal"],
            observation["desired_goal"],
            self.info,
        )
        
        if self.flat_all:
            observation = self._build_obs(observation)
        else:
            observation = self._flatten_obs(observation)
        # plt.imshow(observation['desired_goal'][0])
        # plt.show()
        # input()
        
        is_done = t >= task.EPISODE_LENGTH

        return observation, reward, is_done, self.info

    def reset(self):
        # cannot reset multiple times
        if not self.sim and self.platform is not None:
            raise RuntimeError(
                "Once started, this environment cannot be reset."
            )
        
        if self.sim:
            # hard-reset simulation
            del self.platform
            
            self.platform = trifinger_simulation.TriFingerPlatform(
                visualization=self.visualization,
                object_type=trifinger_simulation.trifinger_platform.ObjectType.DICE,
                enable_cameras=self.enable_cameras,
                # config_dir=SIM_CONFIG_DIR,
                # calib_filename_pattern=SIM_CALIB_FILENAME_PATTERN
            )
        else:
            self.platform = robot_fingers.TriFingerPlatformFrontend()

        # if no goal is given, sample one randomly
        if self.provided_goal is None:
            self.goal = task.sample_goal()
        else:
            self.goal = self.provided_goal
        
        if self.enable_cameras:
            self.goal_masks = task.generate_goal_mask(self.camera_params, self.goal) # TODO: visualize
            
        # visualize the goal
        if self.visualization and self.sim:
            self.goal_markers = []
            for g in self.goal:
                goal_marker = trifinger_simulation.visual_objects.CubeMarker(
                    width=task.DIE_WIDTH,
                    position=g,
                    orientation=(0, 0, 0, 1),
                    pybullet_client_id=self.platform.simfinger._pybullet_client_id,
                )
                self.goal_markers.append(goal_marker)

        self.info = {"time_index": -1}
        self._prev_dice_pos = None
        self._prev_dice_orient = None

        # need to already do one step to get initial observation
        observation, _, _, _ = self.step(self._initial_action, initial=True)

        return observation



class RealRobotRearrangeDiceEnv(gym.GoalEnv):
    """Gym environment for rearranging dice with a TriFingerPro robot."""

    def __init__(
        self,
        goal: typing.Optional[task.Goal] = None,
        action_type: ActionType = ActionType.TORQUE,
        step_size: int = 1,
    ):
        """Initialize.
        Args:
            goal: Goal pattern for the dice.  If ``None`` a new random goal is
                sampled upon reset.
            action_type: Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size:  Number of actual control steps to be performed in one
                call of step().
        """
        # Basic initialization
        # ====================

        if goal is not None:
            task.validate_goal(goal)
        self.goal = goal

        self.action_type = action_type

        if step_size < 1:
            raise ValueError("step_size cannot be less than 1.")
        self.step_size = step_size

        # will be initialized in reset()
        self.platform = None

        # load camera parameters
        self.camera_params = load_camera_parameters(
            CONFIG_DIR, "camera{id}_cropped_and_downsampled.yml"
        )

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        mask_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 270, 270), dtype=np.uint8
        )

        if self.action_type == ActionType.TORQUE:
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")

        self.observation_space = gym.spaces.Dict(
            {
                "robot_observation": gym.spaces.Dict(
                    {
                        "position": robot_position_space,
                        "velocity": robot_velocity_space,
                        "torque": robot_torque_space,
                    }
                ),
                "action": self.action_space,
                "desired_goal": mask_space,
                "achieved_goal": mask_space,
            }
        )

    def compute_reward(
        self,
        achieved_goal: typing.Sequence[np.ndarray],
        desired_goal: typing.Sequence[np.ndarray],
        info: dict,
    ) -> float:
        """Compute the reward for the given achieved and desired goal.
        Args:
            achieved_goal: Segmentation mask of the observed camera images.
            desired_goal: Segmentation mask of the goal positions.
            info: Unused.
        Returns:
            The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal. Note that the following should always hold true::
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'],
                    ob['desired_goal'],
                    info,
                )
        """
        return -task.evaluate_state(desired_goal, achieved_goal)

    def seed(self, seed=None):
        """Sets the seed for this env’s random number generator.
        .. note::
           Spaces need to be seeded separately.  E.g. if you want to sample
           actions directly from the action space using
           ``env.action_space.sample()`` you can set a seed there using
           ``env.action_space.seed()``.
        Returns:
            List of seeds used by this environment.  This environment only uses
            a single seed, so the list contains only one element.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        task.seed(seed)
        return [seed]

    def _create_observation(self, t, action):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)

        segmentation_masks = [
            segment_image(convert_image(c.image))
            for c in camera_observation.cameras
        ]

        observation = {
            "robot_observation": {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
            },
            "action": action,
            "desired_goal": self.goal_masks,
            "achieved_goal": segmentation_masks,
        }
        return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Important: ``reset()`` needs to be called before doing the first step.
        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).
        Returns:
            tuple:
            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.step_size

        # ensure episode length is not exceeded due to step_size
        step_count_after = self.info["time_index"] + num_steps
        if step_count_after > task.EPISODE_LENGTH:
            excess = step_count_after - task.EPISODE_LENGTH
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            self.info["time_index"] = t

            observation = self._create_observation(t, action)

            reward += self.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.info,
            )

            # make sure to not exceed the episode length
            if t >= task.EPISODE_LENGTH - 1:
                break

        is_done = t >= task.EPISODE_LENGTH

        return observation, reward, is_done, self.info

    def reset(self):
        # cannot reset multiple times
        if self.platform is not None:
            raise RuntimeError(
                "Once started, this environment cannot be reset."
            )

        self.platform = robot_fingers.TriFingerPlatformFrontend()

        # if no goal is given, sample one randomly
        if self.goal is None:
            goal = task.sample_goal()
        else:
            goal = self.goal

        self.goal_masks = task.generate_goal_mask(self.camera_params, goal)

        self.info = {"time_index": -1}

        # need to already do one step to get initial observation
        # TODO disable frameskip here?
        observation, _, _, _ = self.step(self._initial_action)

        return observation