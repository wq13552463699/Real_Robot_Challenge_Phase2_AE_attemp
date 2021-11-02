#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:27:41 2021

@author: qiang
"""
# import cv2
# import trifinger_simulation.tasks.rearrange_dice as task
# def preprocess_observation_(observation, bit_depth):
#   observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
#   observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)

# # Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
# def postprocess_observation(observation, bit_depth):
#   return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)

# def _images_to_observation(images, bit_depth):
#   # print(images)
#   # images = torch.tensor(images, dtype=torch.float32)  # Resize and put channel first
#   images = torch.tensor(cv2.resize(images, (270,270), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
#   preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
#   return images.unsqueeze(dim=0)  # Add batch dimension


# from rrc_example_package import rearrange_dice_env_GAN
# env = rearrange_dice_env_GAN.SimtoRealRearrangeDiceEnv(enable_cameras=True,visualization=False)

# import time
# import random
# import os
# from torch.nn import functional as F
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.optim as optim
# import torchvision.utils as vutils
# from torch.autograd import Variable
# from rrc_example_package.scripts.utils import lineplot
# # from nvidia.dali.plugin.pytorch import DALIGenericIterator
# from trifinger_simulation.camera import load_camera_parameters
# # from data import ImagePipeline
# from rrc_example_package.scripts import RRC
# # from torchsummary import summary
# import pathlib
# from copy import copy
# import torchvision.utils as vutils
# import time
# SIM_CONFIG_DIR = pathlib.Path("src/rrc_example_package/camera_params")

# camera_params = load_camera_parameters(
#             SIM_CONFIG_DIR, "camera{id}_cropped_and_downsampled.yml"
#         )

# np.random.seed(42)
# random.seed(10)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.manual_seed(999)

# metrics = {'steps': [],'observation_loss': []}

# gpu_id = 0
# device = torch.device("cuda", gpu_id)

# Encoder = RRC.Encoder().to(device)
# Encoder.apply(RRC.weights_init)
# # summary(Encoder,input_size=(3,270,270))
# Decoder = RRC.Decoder().to(device)
# Decoder.apply(RRC.weights_init)

# optimizerEncoder = optim.Adam(Encoder.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)
# optimizerDecoder = optim.Adam(Decoder.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)


# try:
#     os.mkdir('RRCGAN')
# except OSError:
#     pass

# results_dir = './RRCGAN'

# start_time = time.time()

# # Let's train for 30 epochs (meaning, we go through the entire training set 30 times):

# t = 0
# for _ in range(1000000):
    
#     observation = env.reset()
    
#     for _ in range(50):
        
#         t+=1
        
#         action = env.action_space.sample()
        
#         observation,_,_,_= env.step(action)
        
#         env_pos_mask = np.array(observation["achieved_goal"])
#         env_pos_mask = env_pos_mask.transpose(1, 2, 0)
#         env_pos_mask = _images_to_observation(env_pos_mask ,5)
#         env_pos_mask = env_pos_mask.to(device)
#         latent = Encoder(env_pos_mask)
#         output = Decoder(latent)
        
#         err = F.mse_loss(env_pos_mask, output, reduction='none').sum().mean()#data.cpu().numpy()
        
#         err.backward()
#         optimizerEncoder.step()
#         optimizerDecoder.step()
        
        
#         # print(t2-t1)
#         # print(t)
#         if t % 200 ==0:
#             print(err)
#             # print('[%d/50000] Training absolute losses: L1 %.7f ; L2 %.7f BCE %.7f ; LD %.7f' % ((epoch + t), loss_L1, loss_L2, loss_gan, loss_D,))
#         if t % 3000 ==0: #and t != 0:
#             obs_err = F.mse_loss(env_pos_mask, output, reduction='none').sum().mean().data.cpu().numpy()
#             # print("errerrerrerrerrerrerrerr : ",obs_err)
#             metrics['observation_loss'].append(obs_err)
#             metrics['steps'].append(t)
#             lineplot(metrics['steps'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
#             # Save the inputs, outputs, and ground truth frontals to files:
#             # image1 = env_pos_mask.transpose(1,2,0)
#             # image2 = target_generated_mask.transpose(1,2,0)
#             vutils.save_image(env_pos_mask, 'RRCGAN/%03d_input.jpg' % (t), normalize=True)
#             vutils.save_image(output, 'RRCGAN/%03d_real.jpg' % (t), normalize=True)
    
            
#             # Save the pre-trained Generator as well
#         if t % 20000 == 0:
#             tempresults_dir = './RRCGAN/%s%d' % ('t',t)
#             try:
#                 os.mkdir(tempresults_dir)
#             except OSError:
#                 pass
#             torch.save(Encoder,tempresults_dir+'/Encoder_%d.pt' % (t))
#             torch.save(Decoder,tempresults_dir+'/Decoder_%d.pt' % (t))
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
# Where is your training dataset at?
# datapath = 'training_set'

# You can also choose which GPU you want your model to be trained on below:
gpu_id = 0
device = torch.device("cuda", gpu_id)

# train_pipe = ImagePipeline(datapath, image_size=128, random_shuffle=True, batch_size=30, device_id=gpu_id)
# train_pipe.build()
# m_train = train_pipe.epoch_size()
# print("Size of the training set: ", m_train)
# train_pipe_loader = DALIGenericIterator(train_pipe, ["profiles", "frontals"], m_train)

# Generator:
Encoder = networkRGAN.Encoder().to(device)
Encoder.apply(networkRGAN.weights_init)
Decoder = networkRGAN.Decoder().to(device)
Decoder.apply(networkRGAN.weights_init)
# RNN = networkRGAN.Rnn().to(device)
# RNN.apply(networkRGAN.weights_init)
# Encoder = torch.load("./Encoder_100000.pt")
# Decoder = torch.load("./Decoder_100000.pt")
# Dec = torch.load("./Dec_100000.pt")
# Inc = torch.load("./Inc_100000.pt")


Dec = networkRGAN.Dec().to(device)
Dec.apply(networkRGAN.weights_init)
Inc = networkRGAN.Inc().to(device)
Inc.apply(networkRGAN.weights_init)
# Dec2 = networkRGAN.Dec2().to(device)
# Dec2.apply(networkRGAN.weights_init)
# Inc2 = networkRGAN.Inc2().to(device)
# Inc2.apply(networkRGAN.weights_init)


# netG = network.G().to(device)
# netG.apply(network.weights_init)
# netG = torch.load("./output/netG_11300.pt")
# summary(netG,input_size=(3,128,128))

# Discriminator:
# Dis = networkRGAN.Dis().to(device)
# Dis.apply(networkRGAN.weights_init)

# Here is where you set how important each component of the loss function is:
L1_factor = 0
L2_factor = 1
GAN_factor = 0.1
# GAN factor was decreased to 0.03

criterion = nn.BCELoss() # Binary cross entropy loss

# Optimizers for the generator and the discriminator (Adam is a fancier version of gradient descent with a few more bells and whistles that is used very often):
# optimizerDis = optim.Adam(Dis.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerEncoder = optim.Adam(Encoder.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)
optimizerDecoder = optim.Adam(Decoder.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)
optimizerDec = optim.Adam(Dec.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)
optimizerInc = optim.Adam(Inc.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)
# optimizerDec2 = optim.Adam(Dec2.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)
# optimizerInc2 = optim.Adam(Inc2.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)
# optimizerRNN = optim.Adam(RNN.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)


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
        # Lets keep track of the loss values for each epoch:
        # loss_L1 = 0
        # loss_L2 = 0
        # loss_gan = 0
        
        # Your train_pipe_loader will load the images one batch at a time
        # The inner loop iterates over those batches:
        
        # for i, data in enumerate(train_pipe_loader, 0):
            
            # These are your images from the current batch:
            # profile = data[0]['profiles']
            # frontal = data[0]['frontals']
        
        # hx = torch.randn(1,1, 1024).to(device)
        # for _ in range(20):
        t += 1
        action = env.action_space.sample()
        
        observation,_,_,_= env.step(action)
    
        env_pos_mask = np.array(observation["achieved_goal"])
        env_pos_mask = env_pos_mask.transpose(1, 2, 0)
        env_pos_mask = _images_to_observation(env_pos_mask ,5)
        
        env_pos2 = copy(env_pos_mask)
    
        # TRAINING THE DISCRIMINATOR
        # Dis.zero_grad()
        # real = Variable(env_pos_mask).type('torch.FloatTensor').to(device)
        # target = Variable(torch.ones(real.size()[0])).to(device)
        # output = Dis(real)
        # print(output)
        # D should accept the GT images
        # errD_real = criterion(output, target)
    
        Encoder.zero_grad()
        Decoder.zero_grad()
        Dec.zero_grad()
        Inc.zero_grad()
        # Dec2.zero_grad()
        # Inc2.zero_grad()
        
        
        predict = Variable(env_pos2).type('torch.FloatTensor').to(device)
        generated = Encoder(predict)
        generated = Dec(generated)
        # generated = Dec2(generated)
        # generated = Inc2(generated)
        generated = Inc(generated)
        # generated,hx = RNN(generated.detach(),hx)
        # print(generated.shape)
        generated = Decoder(generated)
        # print('size: ',generated.size())
        # target = Variable(torch.zeros(real.size()[0])).to(device)
        # output = Dis(generated.detach()) # detach() because we are not training G here
    
        # D should reject the synthetic images
        # errD_fake = criterion(output, target)
    
        # errD = errD_real + errD_fake
        # errD.backward()
        # Update D
        # optimizerDis.step()
        
        # TRAINING THE GENERATOR
        # Encoder.zero_grad()
        # Decoder.zero_grad()
        # RNN.zero_grad()
        # 
        # target = Variable(torch.ones(real.size()[0])).to(device)
        # output = Dis(generated)
    
        # G wants to :
        # (a) have the synthetic images be accepted by D (= look like frontal images of people)
        # errG_GAN = criterion(output, target)
    
        # (b) have the synthetic images resemble the ground truth frontal image
        # errG_L1 = torch.mean(torch.abs(real - generated))
        # errG_L2 = torch.mean(torch.pow((real - generated), 2))
        # predict = Variable(env_pos_mask).type('torch.FloatTensor').to(device)
        obs_err = F.mse_loss(predict, generated, reduction='none').sum().mean()#data.cpu().numpy()
    
        # errG = GAN_factor * errG_GAN + L1_factor * errG_L1 + L2_factor * errG_L2
        # print(GAN_factor * errG_GAN)
        # errG = (GAN_factor * errG_GAN) + (obs_err * 0.1)
    
        # loss_L1 = errG_L1.item()
        # loss_L2 = errG_L2.item()
        # loss_gan = errG_GAN.item()
        # loss_D = errD.item()
    
        obs_err.backward()
        # Update G
        optimizerEncoder.step()
        optimizerDecoder.step()
        optimizerDec.step()
        optimizerInc.step()
        # optimizerDec2.step()
        # optimizerInc2.step()
        # optimizerRNN.step()
    
    # if epoch == 0:
    #     print('First training epoch completed in ',(time.time() - start_time),' seconds')
    
    # reset the DALI iterator
    # train_pipe_loader.reset()
    
    # Print the absolute values of three losses to screen:
        if t % 10 ==0:
            print(obs_err)
            # print(errG)
            # print('[%d/50000] Training absolute losses: L1 %.7f ; L2 %.7f BCE %.7f ; LD %.7f' % ((epoch + t), loss_L1, loss_L2, loss_gan, loss_D,))
        
        if t % 100 ==0: #and t != 0:
            obs_err = F.mse_loss(predict, generated, reduction='none').sum().mean().data.cpu().numpy()
            # print("errerrerrerrerrerrerrerr : ",obs_err)
            metrics['observation_loss'].append(obs_err)
            metrics['steps'].append(t)
            lineplot(metrics['steps'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
            # Save the inputs, outputs, and ground truth frontals to files:
            # vutils.save_image(predict.data, 'RNNGAN-bigger/%03d_input.jpg' % (t), normalize=True)
            vutils.save_image(env_pos2.data, 'RNNGAN-512latent/%03d_real.jpg' % (t), normalize=True)
            vutils.save_image(generated.data, 'RNNGAN-512latent/%03d_generated.jpg' % (t), normalize=True)
            
            # Save the pre-trained Generator as well
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
            # torch.save(Dis,tempresults_dir+'/Dis_%d.pt' % (t))
        