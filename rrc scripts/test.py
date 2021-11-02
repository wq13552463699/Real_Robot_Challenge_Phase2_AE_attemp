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
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import os
import numpy as np
import torch.nn.parallel
import torchvision.utils as vutils
from torchvision.utils import save_image
from rrc_example_package.scripts.utils import lineplot
from torchvision import datasets
import cv2
from rrc_example_package import rearrange_dice_env_GAN
from copy import copy
from rrc_example_package.scripts.vae_network import VAE
from rrc_example_package.scripts import ae_network
from torch.autograd import Variable
from torchvision import transforms
from random import randint
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
import torchvision.transforms as T

def loss_fn(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)
    BCE = F.mse_loss(recon_x, x, reduction='none').sum().mean()
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

# def preprocess_observation_(observation, bit_depth):
#   observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
#   observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)

# # Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
# def postprocess_observation(observation, bit_depth):
#   return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)

# def _images_to_observation(images, bit_depth):
#   # print(images)
#   images = torch.tensor(cv2.resize(images, (270,270), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
#   # images = torch.tensor(cv2.resize(images, (64,64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
#   preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
#   return images.unsqueeze(dim=0)  # Add batch dimension

def preprocess_observation_(observation, bit_depth):
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


def _images_to_observation(images, bit_depth):
  # print(images)
  images = torch.tensor(cv2.resize(images, (128,128), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  # images = cv2.resize(images, (128,128), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)  # Resize and put channel first
   # images = torch.tensor(cv2.resize(images, (64,64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  # preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
  return images.unsqueeze(dim=0)  # Add batch dimension
  # return images


env = rearrange_dice_env_GAN.SimtoRealRearrangeDiceEnv(enable_cameras=True,visualization=False)

gpu_id = 0
device = torch.device("cuda", gpu_id)

Encoder = torch.load('./Encoder_500.pt')
Decoder = torch.load('./Decoder_500.pt')
Dec = torch.load('./Dec_500.pt')
Inc = torch.load('./Inc_500.pt')

# dataset = datasets.ImageFolder(root='./dset', transform=transforms.Compose([
#     transforms.Resize(128),
#     transforms.ToTensor(), 
# ]))


Trans =  T.Compose([T.ToTensor(),T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def _AE(mask):
        mask = np.array(mask)   # TODO: the stucture here is not efficient, need to be improved.
        mask = mask.transpose(1,2,0)
        mask = _images_to_observation(mask, 5)
        # mask = mask.numpy()
        # mask = Trans(mask).to(device)
        # k1 = copy(mask)
        # k1 = k1.unsqueeze(dim=0)
        mask = mask.data
        # mask = mask.unsqueeze(dim=0).to(device)
        
        mask = Variable(mask).type('torch.cuda.FloatTensor').to(device)
        
        k1 = copy(mask)
        mask = Encoder(mask)
        mask = Dec(mask)
        mask = Inc(mask)
        mask = Decoder(mask)
        # mask = mask.cpu().detach().numpy()
        
        return k1, mask
    
try:
    os.mkdir('test8_ach')
except OSError:
    pass

results_dir = './test8_ach'
q = 0
for _ in range(5):
    
    a = 50
    i = 0
    env.reset()
    while i <= a:
        i += 1
        action = env.action_space.sample()
        observation,_,_,_= env.step(action)
        
        # if i % 15 == 0:

        #     env_pos_mask = np.array(observation["achieved_goal"])
        #     env_pos_mask = env_pos_mask.transpose(1, 2, 0)
        #     env_pos_mask = _images_to_observation(env_pos_mask ,5)
        #     vutils.save_image(env_pos_mask.data, 'dataset_one_shot/%03d.jpg' % (q), normalize=True)
        #     q += 1
        #     if q % 10 == 0 :
        #         print(q)
        k1,recon = _AE(observation["achieved_goal"])
        
        compare_x = torch.cat([k1, recon])
        
        save_image(compare_x.data.cpu(), './test8_ach/image_%03d.png' % (i+100))   
        # vutils.save_image(env_pos_mask.data, 'dataset_one_shot/%03d.jpg' % (q), normalize=True)
        print(i)


    # fixed_x = dataset[randint(1, 11)][0].unsqueeze(0).to(device)
    # generated = Encoder(fixed_x)
    # generated = Dec(generated)
    # generated = Inc(generated)
    # recon_x = Decoder(generated)
    # loss = F.mse_loss(recon_x, fixed_x, reduction='none').sum().mean()
    # to_print = loss.item()
    # print(to_print)
    # compare_x = torch.cat([fixed_x, recon_x])
    # save_image(compare_x.data.cpu(), './test4/image_%03d.png' % (q))   
    # q += 1

#

   
