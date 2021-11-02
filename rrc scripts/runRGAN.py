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


env = rearrange_dice_env_GAN.SimtoRealRearrangeDiceEnv(enable_cameras=True,visualization=False)

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

# Generator:
Encoder = networkRGAN.Encoder().to(device)
Encoder.apply(networkRGAN.weights_init)
Decoder = networkRGAN.Decoder().to(device)
Decoder.apply(networkRGAN.weights_init)

Dec = networkRGAN.Dec().to(device)
Dec.apply(networkRGAN.weights_init)
Inc = networkRGAN.Inc().to(device)
Inc.apply(networkRGAN.weights_init)

# Here is where you set how important each component of the loss function is:
L1_factor = 0
L2_factor = 1
GAN_factor = 0.1
# GAN factor was decreased to 0.03

criterion = nn.BCELoss() # Binary cross entropy loss
optimizerEncoder = optim.Adam(Encoder.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)
optimizerDecoder = optim.Adam(Decoder.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)
optimizerDec = optim.Adam(Dec.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)
optimizerInc = optim.Adam(Inc.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)



# Create a directory for the output files
try:
    os.mkdir('RNNGAN-512latent')
except OSError:
    pass

results_dir = './RNNGAN-512latent'

start_time = time.time()



# Let's train for 30 epochs (meaning, we go through the entire training set 30 times):
t = 0
for _ in range(100000):
    
    i = 0
    env.reset()
    while i <= 150:
        
        i += 1

        t += 1
        action = env.action_space.sample()
        
        observation,_,_,_= env.step(action)
    
        env_pos_mask = np.array(observation["achieved_goal"])
        env_pos_mask = env_pos_mask.transpose(1, 2, 0)
        env_pos_mask = _images_to_observation(env_pos_mask ,5)
        
        env_pos2 = copy(env_pos_mask)
  
        Encoder.zero_grad()
        Decoder.zero_grad()
        Dec.zero_grad()
        Inc.zero_grad()
        
        predict = Variable(env_pos2).type('torch.FloatTensor').to(device)
        generated = Encoder(predict)
        generated = Dec(generated)
        # generated = Dec2(generated)
        # generated = Inc2(generated)
        generated = Inc(generated)
        # generated,hx = RNN(generated.detach(),hx)
        # print(generated.shape)
        generated = Decoder(generated)


        obs_err = F.mse_loss(predict, generated, reduction='none').sum().mean()#data.cpu().numpy()

    
        obs_err.backward()

        optimizerEncoder.step()
        optimizerDecoder.step()
        optimizerDec.step()
        optimizerInc.step()

        if t % 10 ==0:
            print(obs_err)
        
        if t % 100 ==0: #and t != 0:
            obs_err = F.mse_loss(predict, generated, reduction='none').sum().mean().data.cpu().numpy()

            metrics['observation_loss'].append(obs_err)
            metrics['steps'].append(t)
            lineplot(metrics['steps'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)

            vutils.save_image(env_pos2.data, 'RNNGAN-512latent/%03d_real.jpg' % (t), normalize=True)
            vutils.save_image(generated.data, 'RNNGAN-512latent/%03d_generated.jpg' % (t), normalize=True)
            
        if t % 2000 == 0:
            tempresults_dir = './RNNGAN-512latent/%s%d' % ('t',t)
            try:
                os.mkdir(tempresults_dir)
            except OSError:
                pass
            torch.save(Encoder,tempresults_dir+'/Encoder_%d.pt' % (t))
            torch.save(Decoder,tempresults_dir+'/Decoder_%d.pt' % (t))
            torch.save(Inc,tempresults_dir+'/Inc_%d.pt' % (t))
            torch.save(Dec,tempresults_dir+'/Dec_%d.pt' % (t))

        
