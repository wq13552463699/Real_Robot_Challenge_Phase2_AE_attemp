"""Example Gym environment for the RRC 2021 Phase 2."""
import enum
import typing

import gym
import numpy as np

import robot_fingers
import trifinger_simulation
import trifinger_simulation.visual_objects
from trifinger_simulation import trifingerpro_limits

import trifinger_simulation.tasks.move_cube_on_trajectory as task

from rrc_example_package import cube_trajectory_env


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


class BaseCubeTrajectoryEnv(gym.GoalEnv):
    """Gym environment for moving cubes with TriFingerPro."""

    def __init__(
        self,
        goal_trajectory: typing.Optional[task.Trajectory] = None,
        action_type: ActionType = ActionType.POSITION,
        step_size: int = 1,
    ):
        """Initialize.

        Args:
            goal_trajectory: Goal trajectory for the cube.  If ``None`` a new
                random trajectory is sampled upon reset.
            action_type: Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size:  Number of actual control steps to be performed in one
                call of step().
        """
        # Basic initialization
        # ====================

        if goal_trajectory is not None:
            task.validate_goal(goal_trajectory)
        self.goal = goal_trajectory

        self.action_type = action_type

        if step_size < 1:
            raise ValueError("step_size cannot be less than 1.")
        self.step_size = step_size

        # will be initialized in reset()
        self.platform = None

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

        object_state_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=trifingerpro_limits.object_position.low,
                    high=trifingerpro_limits.object_position.high,
                ),
                "orientation": gym.spaces.Box(
                    low=trifingerpro_limits.object_orientation.low,
                    high=trifingerpro_limits.object_orientation.high,
                ),
            }
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
                "object_observation": gym.spaces.Dict(
                    {
                        "position": object_state_space["position"],
                        "orientation": object_state_space["orientation"],
                    }
                ),
                "action": self.action_space,
                "desired_goal": object_state_space["position"],
                "achieved_goal": object_state_space["position"],
            }
        )

    def compute_reward(
        self,
        achieved_goal: task.Position,
        desired_goal: task.Position,
        info: dict,
    ) -> float:
        """Compute the reward for the given achieved and desired goal.

        Args:
            achieved_goal: Current position of the object.
            desired_goal: Goal position of the current trajectory step.
            info: An info dictionary containing a field "time_index" which
                contains the time index of the achieved_goal.

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
        # This is just some sanity check to verify that the given desired_goal
        # actually matches with the active goal in the trajectory.
        active_goal = np.asarray(
            task.get_active_goal(
                self.info["trajectory"], self.info["time_index"]
            )
        )
        assert np.all(active_goal == desired_goal), "{}: {} != {}".format(
            info["time_index"], active_goal, desired_goal
        )

        return -task.evaluate_state(
            info["trajectory"], info["time_index"], achieved_goal
        )

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

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
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

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
        object_observation = camera_observation.filtered_object_pose

        active_goal = np.asarray(
            task.get_active_goal(self.info["trajectory"], t)
        )

        observation = {
            "robot_observation": {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
            },
            "object_observation": {
                "position": object_observation.position,
                "orientation": object_observation.orientation,
            },
            "action": action,
            "desired_goal": active_goal,
            "achieved_goal": object_observation.position,
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


class SimCubeTrajectoryEnv(BaseCubeTrajectoryEnv):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        goal_trajectory: typing.Optional[task.Trajectory] = None,
        action_type: ActionType = ActionType.POSITION,
        step_size: int = 1,
        visualization: bool = False,
    ):
        """Initialize.

        Args:
            goal_trajectory: Goal trajectory for the cube.  If ``None`` a new
                random trajectory is sampled upon reset.
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
        """
        super().__init__(
            goal_trajectory=goal_trajectory,
            action_type=action_type,
            step_size=step_size,
        )
        self.visualization = visualization

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

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
        step_count_after = self.step_count + num_steps
        if step_count_after > task.EPISODE_LENGTH:
            excess = step_count_after - task.EPISODE_LENGTH
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > task.EPISODE_LENGTH:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            # update goal visualization
            if self.visualization:
                goal_position = task.get_active_goal(
                    self.info["trajectory"], t
                )
                self.goal_marker.set_state(goal_position, (0, 0, 0, 1))

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            self.info["time_index"] = t + 1

            # Alternatively use the observation of step t.  This is the
            # observation from the moment before action_t is applied, i.e. the
            # result of that action is not yet visible in this observation.
            #
            # When using this observation, the resulting cumulative reward
            # should match exactly the one computed during replay (with the
            # above it will differ slightly).
            #
            # self.info["time_index"] = t

            observation = self._create_observation(
                self.info["time_index"], action
            )

            reward += self.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.info,
            )

        is_done = self.step_count >= task.EPISODE_LENGTH
        
        # print('ROBOT OBS:')
        # print('pos: {}'.format(observation['robot_observation']['position']))
        # print('vel: {}'.format(observation['robot_observation']['velocity']))
        # print('torque: {}'.format(observation['robot_observation']['torque']))
        # print('OBJECT OBS:')
        # print('pos: {}'.format(observation['object_observation']['position']))
        # print('orientation: {}'.format(observation['object_observation']['orientation']))

        return observation, reward, is_done, self.info

    def reset(self):
        """Reset the environment."""
        # hard-reset simulation
        del self.platform

        # initialize simulation
        initial_robot_position = trifingerpro_limits.robot_position.default
        # initialize cube at the centre
        initial_object_pose = task.move_cube.Pose(
            position=task.INITIAL_CUBE_POSITION
        )

        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            initial_object_pose=initial_object_pose,
        )

        # if no goal is given, sample one randomly
        if self.goal is None:
            trajectory = task.sample_goal()
        else:
            trajectory = self.goal

        # visualize the goal
        if self.visualization:
            self.goal_marker = trifinger_simulation.visual_objects.CubeMarker(
                width=task.move_cube._CUBE_WIDTH,
                position=trajectory[0][1],
                orientation=(0, 0, 0, 1),
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )

        self.info = {"time_index": -1, "trajectory": trajectory}

        self.step_count = 0

        return self._create_observation(0, self._initial_action)


class CustomSimCubeEnv():
    # TODO: make subclass of SimEnv??
    # Is step_size too coarse?
    # Can velocities of cube be inferred properly??
    
    def __init__(self, difficulty=1, sparse_rewards=True, step_size=40, distance_threshold=0.02, max_steps=50, visualisation=False):
        
        task.GOAL_DIFFICULTY = difficulty
        
        self.env = cube_trajectory_env.SimCubeTrajectoryEnv(
            goal_trajectory=None,  # passing None to sample a random trajectory
            action_type=cube_trajectory_env.ActionType.TORQUE,
            visualization=visualisation,
            step_size=step_size
        )
        self.sparse_rewards = sparse_rewards
        self.prev_object_obs = None
        self._max_episode_steps = max_steps # 50 * step_size simulator steps
        self.distance_threshold = distance_threshold
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        # self.seed = self.env.seed
        if self.sparse_rewards:
            self.compute_reward = self.compute_sparse_reward
        else:
            self.compute_reward = self.env.compute_reward
        
    def step(self, action):
        observation, reward, is_done, info = self.env.step(action)
        
        observation = self.custom_obs(observation)
        
        reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], None)
        
        info["is_success"] = self.compute_sparse_reward(observation['achieved_goal'], observation['desired_goal'], None) == 0
        
        if info["time_index"] >= self._max_episode_steps * self.env.step_size:
            is_done = True
            
        return observation, reward, is_done, info
        
    def reset(self, difficulty=3):
        task.GOAL_DIFFICULTY = difficulty
        
        observation = self.env.reset()
        # Set previous object obs to current (i.e. stationary)
        self.prev_object_obs = np.concatenate((observation["object_observation"]["position"], observation["object_observation"]["orientation"]))
        return self.custom_obs(observation)
    
    # Concat robot obs and object obs into single array.
    # Also include object obs of prev time to infer velocity
    # TODO: Include last 4 frames/calculate velocity directly??
    def custom_obs(self, observation):
        # state obs includes current robot and current object obs, along with previous object obs (to infer velocities)
        state_obs = self.prev_object_obs
        
        # Concat all current robot and object observations
        for key in observation:
            if key == "action":
                # Don't include action or goals in state obs
                break
            for key2 in observation[key]:
                state_obs = np.concatenate((state_obs, observation[key][key2]))
    
        custom_obs = {
            "observation": state_obs,
            "desired_goal": observation["desired_goal"],
            "achieved_goal": observation["achieved_goal"],
            }
        
        # Set new previous object observation
        self.prev_object_obs = np.concatenate((observation["object_observation"]["position"], observation["object_observation"]["orientation"]))
        
        return custom_obs
    
    def compute_sparse_reward(self, achieved_goal, desired_goal, info): # include info to match fetch gym environments (better for compatibility with HER code)
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > self.distance_threshold).astype(np.float32)
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    

class RealRobotCubeTrajectoryEnv(BaseCubeTrajectoryEnv):
    """Gym environment for moving cubes with real TriFingerPro."""

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

        self.platform = robot_fingers.TriFingerPlatformWithObjectFrontend()

        # if no goal is given, sample one randomly
        if self.goal is None:
            trajectory = task.sample_goal()
        else:
            trajectory = self.goal

        self.info = {"time_index": -1, "trajectory": trajectory}

        # need to already do one step to get initial observation
        # TODO disable frameskip here?
        observation, _, _, _ = self.step(self._initial_action)

        return observation
