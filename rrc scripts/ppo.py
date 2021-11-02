#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:56:33 2021

@author: qiang
"""



from stable_baselines3 import PPO
from rrc_example_package import rearrange_dice_env
import numpy as np



# Parallel environments
TOTAL_TIMESTEPS = 100000000
env = rearrange_dice_env.SimtoRealRearrangeDiceEnv(enable_cameras=True,visualization=False,flat_all = True,AE = True)
model = PPO("MlpPolicy", env, verbose=1)


batch = 0
while batch < (TOTAL_TIMESTEPS / 500):
    print(batch)
    model.learn(total_timesteps=500)
    batch += 1 
    model.save("ppo_"+str(batch*500))


# i = 0
# env.reset()
# while i <= 150:
#     i += 1
#     action = env.action_space.sample()
#     observation,_,_,_= env.step(action)
    
#     # if i % 15 == 0:

#     #     env_pos_mask = np.array(observation["achieved_goal"])
#     #     env_pos_mask = env_pos_mask.transpose(1, 2, 0)
#     #     env_pos_mask = _images_to_observation(env_pos_mask ,5)
#     #     vutils.save_image(env_pos_mask.data, 'dataset_one_shot/%03d.jpg' % (q), normalize=True)
#     #     q += 1
#     #     if q % 10 == 0 :
#     #         print(q)
    
#     print(observation.shape)
