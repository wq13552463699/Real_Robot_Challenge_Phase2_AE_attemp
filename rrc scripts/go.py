#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:27:41 2021

@author: qiang
"""

import cv2
from rrc_example_package import rearrange_dice_env_GAN
import trifinger_simulation.tasks.rearrange_dice as task
def preprocess_observation_(observation, bit_depth):
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)

# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)

def _images_to_observation(images, bit_depth):
  # print(images)
  images = torch.tensor(cv2.resize(images, (270,270), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  # images = torch.tensor(cv2.resize(images, (64,64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
  return images.unsqueeze(dim=0)  # Add batch dimension


env = rearrange_dice_env_GAN.SimtoRealRearrangeDiceEnv(enable_cameras=True,visualization=True)

# from __future__ import print_function
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

# from data import ImagePipeline
from rrc_example_package.scripts import networkRGAN
from copy import copy

np.random.seed(42)
random.seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(999)

metrics = {'steps': [],'observation_loss': []}

# You can also choose which GPU you want your model to be trained on below:
gpu_id = 0
device = torch.device("cuda", gpu_id)

Encoder = torch.load("./Encoder_100000.pt")
Decoder = torch.load("./Decoder_100000.pt")
Dec = torch.load("./Dec_100000.pt")
Inc = torch.load("./Inc_100000.pt")


criterion = nn.BCELoss() # Binary cross entropy loss

# Create a directory for the output files
try:
    os.mkdir('RNNGAN-test')
except OSError:
    pass

results_dir = './RNNGAN-test'



# Let's train for 30 epochs (meaning, we go through the entire training set 30 times):
t = 0
env.reset()
for _ in range(100000):
    
    t += 1
    action = env.action_space.sample()
    
    observation,_,_,_= env.step(action)
    
    print(t)
    if t % 50 ==0: #and t != 0:
        
        env_pos_mask = np.array(observation["achieved_goal"])
        env_pos_mask = env_pos_mask.transpose(1, 2, 0)
        env_pos_mask = _images_to_observation(env_pos_mask ,5)
        
        predict = Variable(env_pos_mask).type('torch.FloatTensor').to(device)
        generated = Encoder(predict)
        generated = Dec(generated)
        generated = Inc(generated)
        generated = Decoder(generated)
        
        vutils.save_image(env_pos_mask.data, 'RNNGAN-test/%03d_real.jpg' % (t), normalize=True)
        vutils.save_image(generated.data, 'RNNGAN-test/%03d_generated.jpg' % (t), normalize=True)
        
