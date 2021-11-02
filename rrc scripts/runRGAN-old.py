#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:27:41 2021

@author: qiang
"""
import cv2
import trifinger_simulation.tasks.rearrange_dice as task
def preprocess_observation_(observation, bit_depth):
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)

# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)

def _images_to_observation(images, bit_depth):
  # print(images)
  # images = torch.tensor(images, dtype=torch.float32)  # Resize and put channel first
  images = torch.tensor(cv2.resize(images, (270,270), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
  return images.unsqueeze(dim=0)  # Add batch dimension


from rrc_example_package import rearrange_dice_env_GAN
env = rearrange_dice_env_GAN.SimtoRealRearrangeDiceEnv(enable_cameras=True,visualization=True)

import time
import random
import os
from torch.nn import functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from rrc_example_package.scripts.utils import lineplot
# from nvidia.dali.plugin.pytorch import DALIGenericIterator
from trifinger_simulation.camera import load_camera_parameters
# from data import ImagePipeline
from rrc_example_package.scripts import RRC
# from torchsummary import summary
import pathlib
from copy import copy

SIM_CONFIG_DIR = pathlib.Path("src/rrc_example_package/camera_params")

camera_params = load_camera_parameters(
            SIM_CONFIG_DIR, "camera{id}_cropped_and_downsampled.yml"
        )

np.random.seed(42)
random.seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(999)

metrics = {'steps': [],'observation_loss': []}

gpu_id = 0
device = torch.device("cuda", gpu_id)

Encoder = RRC.Encoder().to(device)
Encoder.apply(RRC.weights_init)
# summary(Encoder,input_size=(3,270,270))

optimizerEncoder = optim.Adam(Encoder.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)

try:
    os.mkdir('RRCGAN')
except OSError:
    pass

results_dir = './RRCGAN'

start_time = time.time()

# Let's train for 30 epochs (meaning, we go through the entire training set 30 times):
observation = env.reset()
t = 0
for _ in range(10000000):
    
    t+=1
    
    action = env.action_space.sample()
    
    observation,_,_,_= env.step(action)
    
    env_pos_mask = np.array(observation["achieved_goal"])
    env_pos_mask = env_pos_mask.transpose(1, 2, 0)
    env_pos_mask = _images_to_observation(env_pos_mask ,5)
    env_pos_mask = env_pos_mask.to(device)
    env_pos_mask = Variable(env_pos_mask,requires_grad=True).type('torch.cuda.FloatTensor')
    print(env_pos_mask.shape)
    generated = Encoder(env_pos_mask)
    
    # generated = generated.detach().numpy()
    generated = generated.detach().cpu().numpy()
    print(1)
    target_generated_mask = np.array(task.generate_goal_mask(camera_params, generated))
    print(2)
    target_generated_mask = target_generated_mask.transpose(1, 2, 0)
    print(3)
    target_generated_mask = _images_to_observation(target_generated_mask,5).to(device)
    print(4)
    err = F.mse_loss(env_pos_mask, target_generated_mask, reduction='none').sum().mean()#data.cpu().numpy()
    print(5)
    err.backward()
    print(6)
    optimizerEncoder.step()
    
    
    print(t)
    if t % 1000 ==0:
        print(err)
        # print('[%d/50000] Training absolute losses: L1 %.7f ; L2 %.7f BCE %.7f ; LD %.7f' % ((epoch + t), loss_L1, loss_L2, loss_gan, loss_D,))
    if t % 3000 ==0: #and t != 0:
        obs_err = F.mse_loss(env_pos_mask, target_generated_mask, reduction='none').sum().mean().data.cpu().numpy()
        # print("errerrerrerrerrerrerrerr : ",obs_err)
        metrics['observation_loss'].append(obs_err)
        metrics['steps'].append(t)
        lineplot(metrics['steps'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
        # Save the inputs, outputs, and ground truth frontals to files:
        image1 = env_pos_mask.transpose(1,2,0)
        image2 = target_generated_mask.transpose(1,2,0)
        cv2.imwrite(image1, 'RRCGAN/%03d_input.jpg' % (t), normalize=True)
        cv2.imwrite(image2, 'RRCGAN/%03d_real.jpg' % (t), normalize=True)

        
        # Save the pre-trained Generator as well
    if t % 5000 == 0:
        tempresults_dir = './RRCGAN/%s%d' % ('t',t)
        try:
            os.mkdir(tempresults_dir)
        except OSError:
            pass
        torch.save(Encoder,tempresults_dir+'/Encoder_%d.pt' % (t))
# from trifinger_simulation.camera import load_camera_parameters
# SIM_CONFIG_DIR = pathlib.Path("src/rrc_example_package/camera_params")
# camera_params = load_camera_parameters(
#             SIM_CONFIG_DIR, "camera{id}_cropped_and_downsampled.yml"
#         )
# goal = [(0.10999999999999999, -0.13199999999999998, 0.1), (-0.022000000000000002, -0.088,0.1), 
#  (-0.022000000000000002, 0.176, 0.05), (-0.044, 0.088, 0.05), 
#  (0.06599999999999999, -0.022000000000000002, 0.011), (0.044, 0.132, 0.011), 
#  (0.022, -0.088, 0.05), (0.044, -0.13199999999999998, 0.05), (0.0, 0.088,0.05), 
#  (0.022, 0.044, 0.15), (0.0, -0.13199999999999998, 0.15), (0.022, 0.154, 0.15), 
#  (-0.066, 0.10999999999999999, 0.01), (0.154, 0.06599999999999999, 0.01), (-0.15399999999999997, 0.0, 0.1),
#  (-0.044, 0.022, 0.011), (0.132, 0.0, 0.011), (-0.066, -0.022000000000000002, 0.011), (0.0, 0.176, 0.011), 
#  (-0.13199999999999998, 0.044, 0.1), (-0.13199999999999998, -0.044, 0.2), (-0.066, 0.022, 0.2), (-0.11, 0.0, 0.2),
#  (-0.022000000000000002, -0.11, 0.2), (-0.044, 0.132, 0.05)]

# goal_masks = task.generate_goal_mask(camera_params, goal)

# image = np.array(goal_masks)
# image = image.transpose(1, 2, 0)
# # image = cv2.resize(image, (64,64), interpolation=cv2.INTER_LINEAR)
# (B, G, R) = cv2.split(image)
# cv2.imwrite("B.jpg", B)
# cv2.imwrite("G.jpg", G)
# cv2.imwrite("R.JPG", R)
# (B, G, R) = cv2.split(image)
# cv2.imwrite("B1.jpg", B)
# cv2.imwrite("G1.jpg", G)
# cv2.imwrite("R1.JPG", R)

