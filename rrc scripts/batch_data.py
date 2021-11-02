#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 18:37:18 2021

@author: qiang
"""

import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
# import imutils
import itertools
from torchvision.utils import save_image
import torchvision.models as models
import torch
from torchvision import transforms
from torch.nn import functional as F
from rrc_example_package import rearrange_dice_env
import trifinger_simulation.tasks.rearrange_dice as task
from trifinger_object_tracking.py_lightblue_segmenter import segment_image
import torch.nn as nn
from rrc_example_package.scripts import batch_network
from torch.autograd import Variable
from copy import copy


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


from rrc_example_package.her.mpi_utils.simple_rms import RunningMeanStd


coord_norm = RunningMeanStd(shape=(75))

FACE_CORNERS = (
    (0, 1, 2, 3),
    (4, 5, 1, 0),
    (5, 6, 2, 1),
    (7, 6, 2, 3),
    (4, 7, 3, 0),
    (4, 5, 6, 7),
)

Encoder = torch.load('./Encoder_5000.pt')
Decoder = torch.load('./Decoder_5000.pt')
Dec = torch.load('./Dec_5000.pt')
Inc = torch.load('./Inc_5000.pt')
optimizerEncoder = optim.Adam(Encoder.parameters(), lr = 0.00015, betas = (0.5, 0.999), eps = 1e-8)
optimizerDec = optim.Adam(Dec.parameters(), lr = 0.00015, betas = (0.5, 0.999), eps = 1e-8)

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

gpu_id = 0
device = torch.device("cuda", gpu_id)

def generate_batch(env, batch_size):
    batch = np.ones((batch_size, 384))
    goals = np.ones((batch_size, 25 * 3))
    for i in range(batch_size):
        seg_mask = np.ones((3, 270, 270))
        g = task.sample_goal()
        goal = list(itertools.chain(*g))
		#goal = [g for i, g in enumerate(goal) if ((i+1) % 3) !=0]
        goals[i] = np.array(goal)
        for idx, c in enumerate(env.camera_params):
            seg_mask[idx,:,:] = generate_goal_mask(c, g)
        seg_mask = seg_mask.transpose(1, 2, 0)
        seg_mask =  cv2.resize(seg_mask, (128,128), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        seg_mask /= 255.0
        
        seg_mask = torch.from_numpy(seg_mask).unsqueeze(dim=0)
        seg_mask = Variable(seg_mask).type('torch.cuda.FloatTensor').to(device)
        

        # k1 = copy(seg_mask)
        seg_mask = Encoder(seg_mask)
        seg_mask = Dec(seg_mask)
        # seg_mask = Inc(seg_mask)
        # seg_mask = Decoder(seg_mask)
        # compare_x = torch.cat([k1, seg_mask])
        # save_image(compare_x.data.cpu(), './batch_image_%03d.png' % (i))   
 
        batch[i] = seg_mask.cpu().detach().numpy()
# 		batch[i] = torch.tensor(cv2.resize(images, (128,128), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
# 		print(seg_mask.shape)
# 		a = seg_mask.transpose(1,2,0)
# 		cv2.imwrite('%d.jpg' % (i),a)
    return batch, goals

##########################################################################################################################
# def generate_batch(env, batch_size):
#  	batch = np.ones((batch_size, 3, 128, 128))
#  	goals = np.ones((batch_size, 25 * 3))
#  	#goals = np.ones((batch_size, 25 * 2))
#  	for i in range(batch_size):
# 		seg_mask = np.ones((3, 270, 270))
# 		g = task.sample_goal()
# 		goal = list(itertools.chain(*g))
# 		#goal = [g for i, g in enumerate(goal) if ((i+1) % 3) !=0]
# 		goals[i] = np.array(goal)
# 		for idx, c in enumerate(env.camera_params):
#  			seg_mask[idx,:,:] = generate_goal_mask(c, g)
# 		#segmentation_masks = np.array([segment_image(cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR)) for c in obs.cameras])
# 		seg_mask = seg_mask.transpose(1, 2, 0)
# 		seg_mask =  cv2.resize(seg_mask, (128,128), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
# 		seg_mask /= 255.0
# 		batch[i] = seg_mask
        
#  	return batch, goals
##########################################################################################################################


def get_cell_corners_3d(pos):
    """Get 3d positions of the corners of the cell at the given position."""
    d = 0.022 / 2
    nppos = np.asarray(pos)

    # order of the corners is the same as in the cube model of the
    # trifinger_object_tracking package
    # people.tue.mpg.de/mpi-is-software/robotfingers/docs/trifinger_object_tracking/doc/cube_model.html
    return np.array(
        (
            nppos + (d, -d, d),
            nppos + (d, d, d),
            nppos + (-d, d, d),
            nppos + (-d, -d, d),
            nppos + (d, -d, -d),
            nppos + (d, d, -d),
            nppos + (-d, d, -d),
            nppos + (-d, -d, -d),
        )
    )


def generate_goal_mask(camera_parameters, goal):
    """Generate goal masks that can be used with :func:`evaluate_state`.
    A goal mask is a single-channel image where the areas at which dice are
    supposed to be placed are white and everything else is black.  So it
    corresponds more or less to a segmentation mask where all dice are at the
    goal positions.
    For rendering the mask, :data:`TARGET_WIDTH` is used for the die width to
    add some tolerance.
    Args:
        camera_parameters: List of camera parameters, one per camera.
        goal: The goal die positions.
    Returns:
        List of masks.  The number and order of masks corresponds to the input
        ``camera_parameters``.
    """
    #masks = []
    #for cam in camera_parameters:
    #mask = np.zeros((camera_parameters.image_height, camera_parameters.image_width), dtype=np.uint8)
    mask = np.zeros((270, 270), dtype=np.uint8)

    # get camera position and orientation separately
    tvec = camera_parameters.tf_world_to_camera[:3, 3]
    rmat = camera_parameters.tf_world_to_camera[:3, :3]
    rvec = Rotation.from_matrix(rmat).as_rotvec()

    for pos in goal:
        corners = get_cell_corners_3d(pos)

        # project corner points into the image
        projected_corners, _ = cv2.projectPoints(
            corners,
            rvec,
            tvec,
            camera_parameters.camera_matrix,
            camera_parameters.distortion_coefficients,
        )

        # draw faces in mask
        for face_corner_idx in FACE_CORNERS:
            points = np.array(
                [projected_corners[i] for i in face_corner_idx],
                dtype=np.int32,
            )
            mask = cv2.fillConvexPoly(mask, points, 255)

        #masks.append(mask)

    return mask


def loss_fn(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)
    BCE = F.mse_loss(recon_x, x, reduction='none').sum().mean()
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

##########################################################################################################################
# env =rearrange_dice_env.SimtoRealRearrangeDiceEnv(enable_cameras=True,visualization=False,step_size=1,image_size=270)
# env.reset()


# # def train_AE(env, batch_size):
# metrics = {'steps': [],'observation_loss': []}

# # Encoder = ae_network.Encoder().to(device)
# # Encoder.apply(ae_network.weights_init)
# # Decoder = ae_network.Decoder().to(device)
# # Decoder.apply(ae_network.weights_init)
# # Dec = ae_network.Dec().to(device)
# # Dec.apply(ae_network.weights_init)
# # Inc = ae_network.Inc().to(device)
# # Inc.apply(ae_network.weights_init)

# Encoder = torch.load('./Encoder_5000.pt')
# Decoder = torch.load('./Decoder_5000.pt')
# Dec = torch.load('./Dec_5000.pt')
# Inc = torch.load('./Inc_5000.pt')

# optimizerEncoder = optim.Adam(Encoder.parameters(), lr = 0.00015, betas = (0.5, 0.999), eps = 1e-8)
# optimizerDecoder = optim.Adam(Decoder.parameters(), lr = 0.00015, betas = (0.5, 0.999), eps = 1e-8)
# optimizerDec = optim.Adam(Dec.parameters(), lr = 0.00015, betas = (0.5, 0.999), eps = 1e-8)
# optimizerInc = optim.Adam(Inc.parameters(), lr = 0.00015, betas = (0.5, 0.999), eps = 1e-8)


# bs = 4
# try:
#     os.mkdir('FF3')
# except OSError:
#     pass

# results_dir = './FF3'

# epochs = 5000

# for epoch in range(epochs+1):
    
#     env.reset()
#     input_batch, goals = generate_batch(env, bs)  
#     loss = 0
#     for index, mask in enumerate(input_batch):

#         mask = torch.from_numpy(mask).unsqueeze(dim=0)
#         image = Variable(mask).type('torch.cuda.FloatTensor').to(device)
        
#         generated = Encoder(image)
#         generated = Dec(generated)
#         generated = Inc(generated)
#         recon_x = Decoder(generated)

#         loss += F.mse_loss(recon_x, image, reduction='none').sum().mean()

#     Encoder.zero_grad()
#     Decoder.zero_grad()
#     Dec.zero_grad()
#     Inc.zero_grad()
    
#     loss.backward()
    
#     optimizerEncoder.step()
#     optimizerDecoder.step()
#     optimizerDec.step()
#     optimizerInc.step()

#     to_print = "Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.item()/bs)
#     print(to_print)
    
#     metrics['observation_loss'].append(loss.item()/bs)
#     metrics['steps'].append(epoch)
#     lineplot(metrics['steps'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
    
#     if epoch % 10 == 0:
        
#         mask,coor = generate_batch(env, 1)
#         mask = mask[0]

        
#         mask = torch.from_numpy(mask)
#         mask = mask.unsqueeze(dim=0)
#         mask = Variable(mask).type('torch.cuda.FloatTensor').to(device)
        
#         generated = Encoder(mask)
#         generated = Dec(generated)
#         generated = Inc(generated)
#         recon_x = Decoder(generated)

        
#         compare_x = torch.cat([mask, recon_x])
        
#         save_image(compare_x.data.cpu(), './FF2/image_%03d.png' % (epoch))  

        
#     if epoch % 20 ==0:
#         tempresults_dir = './FF2/%s%d' % ('epoch',epoch)
#         try:
#             os.mkdir(tempresults_dir)
#         except OSError:
#             pass
#         torch.save(Encoder,tempresults_dir+'/Encoder_%d.pt' % (epoch))
#         torch.save(Decoder,tempresults_dir+'/Decoder_%d.pt' % (epoch))
#         torch.save(Inc,tempresults_dir+'/Inc_%d.pt' % (epoch))
#         torch.save(Dec,tempresults_dir+'/Dec_%d.pt' % (epoch))

##########################################################################################################################
# bs = 16
# env =rearrange_dice_env.SimtoRealRearrangeDiceEnv(enable_cameras=True,visualization=False,step_size=1,image_size=270)
# env.reset()
# input_batch, goals = generate_batch(env, bs)  
# print(input_batch.shape)
# print(goals.shape)

metrics = {'steps': [],'observation_loss': []}

Coor = batch_network.Coor().to(device)
Coor.apply(batch_network.weights_init)

optimizerCoor = optim.Adam(Coor.parameters(), lr = 0.00015, betas = (0.5, 0.999), eps = 1e-8)
# optimizerCoor = optim.SGD(Coor.parameters(), lr = 0.000001, momentum=0.6)

try:
    os.mkdir('Coor2')
except OSError:
    pass

results_dir = './Coor2'

epochs = 500000
bs = 32

env =rearrange_dice_env.SimtoRealRearrangeDiceEnv(enable_cameras=True,visualization=False,step_size=1,image_size=270)

coord_norm = RunningMeanStd(shape=(75))
coord_norm1 = RunningMeanStd(shape=(384))


for epoch in range(epochs+1):
    
    env.reset()
    input_batch, goals = generate_batch(env, bs)    
    loss = 0
    for index, mask in enumerate(input_batch):
        
        mask = np.array(mask)
        coord_norm1.update(mask)
        mask = coord_norm1.normalize(mask)
        mask = torch.from_numpy(mask)
        image = Variable(mask).type('torch.cuda.FloatTensor').to(device)
        generated = Coor(image)
        
        goal = np.array(goals[index])
        coord_norm.update(goal)
        goal = coord_norm.normalize(goal)
        # goal = goal * 100
        goal = torch.from_numpy(goal)
        goal = Variable(goal).type('torch.cuda.FloatTensor').to(device)
        
        loss += F.mse_loss(generated, goal, reduction='none').sum().mean()

    Coor.zero_grad()
    
    loss.backward()
    
    optimizerCoor.step()

        
    to_print = "Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.item()/bs)
    print(to_print)
    
    metrics['observation_loss'].append(loss.item()/bs)
    metrics['steps'].append(epoch)
    lineplot(metrics['steps'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
    
    if epoch % 30 == 0:
        mask,coor = generate_batch(env, 1)
        mask = mask[0]
        
        mask = np.array(mask)
        coord_norm1.update(mask)
        mask = coord_norm1.normalize(mask)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(dim=0)
        mask = Variable(mask).type('torch.cuda.FloatTensor').to(device)
        generated = Coor(image)
        
        coor = coor[0]
        # coor = coor * 100
        coor = torch.from_numpy(coord_norm.normalize(coor))
        coor = Variable(coor).type('torch.cuda.FloatTensor').to(device)

        print('coor',coor.data)
        print('generated',generated.data)
        
    if epoch % 500 ==0:
        tempresults_dir = './Coor2/%s%d' % ('epoch',epoch)
        try:
            os.mkdir(tempresults_dir)
        except OSError:
            pass

        torch.save(Coor,tempresults_dir+'/Coor2_%d.pt' % (epoch))

##########################################################################################################################
# import torch.optim as optim
# import os
# import torch
# import torch.nn.functional as F
# import os
# import numpy as np
# import torch.nn.parallel
# import torchvision.utils as vutils
# from torchvision.utils import save_image
# from rrc_example_package.scripts.utils import lineplot
# from torchvision import datasets
# import cv2
# from rrc_example_package import rearrange_dice_env_GAN
# from copy import copy
# from rrc_example_package.scripts.vae_network import VAE
# from rrc_example_package.scripts import ae_network
# from torch.autograd import Variable
# from torchvision import transforms
# from random import randint
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
# from rrc_example_package.her.mpi_utils.simple_rms import RunningMeanStd

# metrics = {'steps': [],'observation_loss': []}






# Decoder = batch_network.Decoder().to(device)
# Decoder.apply(batch_network.weights_init)

# optimizerDecoder = optim.Adam(Decoder.parameters(), lr = 0.00015, betas = (0.5, 0.999), eps = 1e-8)

# try:
#     os.mkdir('DEC_V3')
# except OSError:
#     pass

# results_dir = './DEC_V3'

# epochs = 500000
# bs = 128

# env =rearrange_dice_env.SimtoRealRearrangeDiceEnv(enable_cameras=True,visualization=False,step_size=1,image_size=270)

# coord_norm = RunningMeanStd(shape=(75))


# for epoch in range(epochs+1):
    
#     env.reset()
#     input_batch, goals = generate_batch(env, bs)    
#     loss = 0
#     for index, mask in enumerate(input_batch):
#         mask/=255.0
#         mask = torch.from_numpy(mask)
#         image = Variable(mask).type('torch.cuda.FloatTensor').to(device)
        
#         goal = np.array(goals[index])
#         coord_norm.update(goal)
#         goal = coord_norm.normalize(goal)
#         goal = torch.from_numpy(goal)
#         goal = Variable(goal).type('torch.cuda.FloatTensor').to(device)
#         generated = Decoder(goal)

#         loss += F.mse_loss(generated, image, reduction='none').sum().mean()

#     Decoder.zero_grad()
    
#     loss.backward()
    
#     optimizerDecoder.step()

        
#     to_print = "Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.item()/bs)
#     print(to_print)
    
#     metrics['observation_loss'].append(loss.item()/bs)
#     metrics['steps'].append(epoch)
#     lineplot(metrics['steps'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
    
#     if epoch % 30 == 0:
#         mask,coor = generate_batch(env, 1)
#         mask = mask[0]
#         mask/=255.0
        
#         mask = torch.from_numpy(mask)
#         mask = mask.unsqueeze(dim=0)
#         mask = Variable(mask).type('torch.cuda.FloatTensor').to(device)
        
#         coor = coor[0]
#         coor = torch.from_numpy(coord_norm.normalize(coor))
#         coor = Variable(coor).type('torch.cuda.FloatTensor').to(device)

#         recon_x = Decoder(coor)
        
#         compare_x = torch.cat([mask, recon_x])
        
#         save_image(compare_x.data.cpu(), './DEC_V3/image_%03d.png' % (epoch))   
        
#     if epoch % 2000 ==0:
#         tempresults_dir = './DEC_V3/%s%d' % ('epoch',epoch)
#         try:
#             os.mkdir(tempresults_dir)
#         except OSError:
#             pass

#         torch.save(Decoder,tempresults_dir+'/Decoder_%d.pt' % (epoch))


# optim = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=0.000001)
# min_cost = 10000
# while True:

#     input_batch, goals = generate_batch(env, 16)
#     input_batch = torch.from_numpy(input_batch).float()
#     goals = torch.from_numpy(goals).float()
#     loss = torch.nn.MSELoss()
#     if torch.cuda.is_available():
#         input_batch = input_batch.to('cuda')
#         goals = goals.to('cuda')
#         resnet.to('cuda')

#     """preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])"""

#     #input_batch = preprocess(input_batch)
#     input_batch = torch.nn.functional.normalize(input_batch)
#     out = resnet(input_batch)
#     cost = loss(out, goals)
#     cost.backward()
#     optim.step()
#     print("Loss: {}".format(cost))
#     if cost < min_cost:
#         min_cost = cost
#         torch.save(resnet.state_dict(), './best_model_resnet_2.pth')


# loss = nn.MSELoss()