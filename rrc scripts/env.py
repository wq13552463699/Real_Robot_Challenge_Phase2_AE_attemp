import cv2
import numpy as np
import torch
import gym
from torch.autograd import Variable

# from CoppeliaSim_UR5_gym_v2 import CoppeliaSim_UR5_gym_v2
from rrc_example_package import rearrange_dice_env

# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


def _images_to_observation(images, bit_depth):
  # print(images)
  images = torch.tensor(cv2.resize(images, (64,64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  # images = torch.tensor(cv2.resize(images, (64,64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
  return images.unsqueeze(dim=0)  # Add batch dimension



gpu_id = 0
device = torch.device("cuda", gpu_id)

class GymEnv():
  def __init__(self, symbolic, seed, max_episode_length, action_repeat, bit_depth):
    import logging
    gym.logger.set_level(logging.ERROR)  # Ignore warnings from Gym logger
    self.symbolic = symbolic
    self._env = rearrange_dice_env.SimtoRealRearrangeDiceEnv(enable_cameras=True,visualization=False,flat_all = True,AE = True)
    # self._env = gym.make('MountainCarContinuous-v0')
    self._env.seed(seed)
    self.max_episode_length = max_episode_length
    self.action_repeat = action_repeat
    self.bit_depth = bit_depth

  def reset(self):
    self.t = 0  # Reset internal timer
    state = self._env.reset()
    if self.symbolic:
      return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    else:
      return _images_to_observation(state, self.bit_depth)
  
  def step(self, action):
    action = action.detach().numpy()
    # reward = 0
    for k in range(self.action_repeat):
      state, reward_k, done, _ = self._env.step(action)
      # reward += reward_k
      self.t += 1  # Increment internal timer
      done = done or self.t == self.max_episode_length
      if done:
        break
    
    if self.symbolic:
      # env_pos_mask = np.array(state["achieved_goal"])
      # env_pos_mask = env_pos_mask.transpose(1, 2, 0)
      # env_pos_mask = _images_to_observation(env_pos_mask ,5)
      # predict = Variable(env_pos_mask).type('torch.FloatTensor').to(device)
      # generated = Encoder(predict)
      # generated = Dec(generated)
      # # generated = Dec2(generated)
      # # generated = Inc2(generated)
      # generated = Inc(generated)
      # # generated,hx = RNN(generated.detach(),hx)
      # # print(generated.shape)
      # generated = Decoder(generated)
      
      # a_goal = generated.cpu().numpy()
      
      
      observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    else:
      observation = _images_to_observation(state ,self.bit_depth)
    return observation, reward_k, done


  def render(self):
    self._env.render()

  def close(self):
    self._env.close()

  @property
  def observation_size(self):
    return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

  @property
  def action_size(self):
    return self._env.action_space.shape[0]

  @property
  def action_range(self):
    return float(self._env.action_space.low[0]), float(self._env.action_space.high[0])
# def Env(env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
#   if env in GYM_ENVS:
#     return GymEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)
#   elif env in CONTROL_SUITE_ENVS:
#     return ControlSuiteEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    return torch.from_numpy(self._env.action_space.sample())

# Wrapper for batching environments together
class EnvBatcher():
  def __init__(self, env_class, env_args, env_kwargs, n):
    self.n = n
    self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
    self.dones = [True] * n

  # Resets every environment and returns observation
  def reset(self):
    observations = [env.reset() for env in self.envs]
    self.dones = [False] * self.n
    return torch.cat(observations)

 # Steps/resets every environment and returns (observation, reward, done)
  def step(self, actions):
    done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]  # Done mask to blank out observations and zero rewards for previously terminated environments
    observations, rewards, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
    dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]  # Env should remain terminated if previously terminated
    self.dones = dones
    observations, rewards, dones = torch.cat(observations), torch.tensor(rewards, dtype=torch.float32), torch.tensor(dones, dtype=torch.uint8)
    observations[done_mask] = 0
    rewards[done_mask] = 0
    return observations, rewards, dones

  def close(self):
    [env.close() for env in self.envs]
