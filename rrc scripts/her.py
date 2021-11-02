from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from rrc_example_package import rearrange_dice_env

model_class = DDPG  # works also with SAC, DDPG and TD3
N_BITS = 200

# env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)
env = rearrange_dice_env.SimtoRealRearrangeDiceEnv(enable_cameras=True,visualization=False,flat_all = False,AE = True)

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = N_BITS

# Initialize the model
model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
    ),
    verbose=1,
)
model = model_class.load('./her_bit_env', env=env)
# Train the model
model.learn(1000000000)

model.save("./her_bit_env")

# TOTAL_TIMESTEPS = 100000000
# batch = 0
# while batch < (TOTAL_TIMESTEPS / 200):
#     print(batch)
#     model.learn(total_timesteps=200)
#     batch += 1 
#     if batch % 100 ==0:
#         model.save("ppo_"+str(batch*200))
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
# 

# obs = env.reset()
# for _ in range(100):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, _ = env.step(action)

#     if done:
#         obs = env.reset()