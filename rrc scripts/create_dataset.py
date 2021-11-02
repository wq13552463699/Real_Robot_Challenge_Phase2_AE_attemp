#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 00:58:56 2021

@author: qiang
"""
import torch
import torch.nn.functional as F
import os
import numpy as np
import torch.nn.parallel
import torchvision.utils as vutils
from rrc_example_package.scripts.utils import lineplot
import cv2
from rrc_example_package import rearrange_dice_env_GAN
from copy import copy
from rrc_example_package.scripts.vae_network import VAE
from torch.autograd import Variable

def loss_fn(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)
    BCE = F.mse_loss(recon_x, x, reduction='none').sum().mean()
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

def preprocess_observation_(observation, bit_depth):
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)

# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)

def _images_to_observation(images, bit_depth):
  # print(images)
  images = torch.tensor(cv2.resize(images, (128,128), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  # images = torch.tensor(cv2.resize(images, (64,64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
  return images.unsqueeze(dim=0)  # Add batch dimension


env = rearrange_dice_env_GAN.SimtoRealRearrangeDiceEnv(enable_cameras=True,visualization=False)

# import random

# try:
#     os.mkdir('dataset_one_shot')
# except OSError:
#     pass

# results_dir = './dataset_one_shot'

# q = 23000
# for _ in range(1000):
#     a = random.randint(50, 150)
#     i = 0
#     env.reset()
#     while i <= a:
#         i += 1
#         action = env.action_space.sample()
#         observation,_,_,_= env.step(action)
        
#         # if i % 15 == 0:

#         #     env_pos_mask = np.array(observation["achieved_goal"])
#         #     env_pos_mask = env_pos_mask.transpose(1, 2, 0)
#         #     env_pos_mask = _images_to_observation(env_pos_mask ,5)
#         #     vutils.save_image(env_pos_mask.data, 'dataset_one_shot/%03d.jpg' % (q), normalize=True)
#         #     q += 1
#         #     if q % 10 == 0 :
#         #         print(q)
#     env_pos_mask = np.array(observation["achieved_goal"])
#     env_pos_mask = env_pos_mask.transpose(1, 2, 0)
#     env_pos_mask = _images_to_observation(env_pos_mask ,5)
#     vutils.save_image(env_pos_mask.data, 'dataset_one_shot/%03d.jpg' % (q), normalize=True)
#     q += 1
#     print(q)


            