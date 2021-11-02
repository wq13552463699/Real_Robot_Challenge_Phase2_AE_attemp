#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:27:41 2021

@author: qiang
"""

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

def loss_fn(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)
    BCE = F.mse_loss(recon_x, x, reduction='none').sum().mean()
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD


metrics = {'steps': [],'observation_loss': []}

gpu_id = 0
device = torch.device("cuda", gpu_id)

Encoder = ae_network.Encoder().to(device)
Encoder.apply(ae_network.weights_init)
Decoder = ae_network.Decoder().to(device)
Decoder.apply(ae_network.weights_init)
Dec = ae_network.Dec().to(device)
Dec.apply(ae_network.weights_init)
Inc = ae_network.Inc().to(device)
Inc.apply(ae_network.weights_init)

# Encoder = torch.load('./Encoder_750.pt')
# Decoder = torch.load('./Decoder_750.pt')
# Dec = torch.load('./Dec_750.pt')
# Inc = torch.load('./Inc_750.pt')

optimizerEncoder = optim.Adam(Encoder.parameters(), lr = 0.00015, betas = (0.5, 0.999), eps = 1e-8)
optimizerDecoder = optim.Adam(Decoder.parameters(), lr = 0.00015, betas = (0.5, 0.999), eps = 1e-8)
optimizerDec = optim.Adam(Dec.parameters(), lr = 0.00015, betas = (0.5, 0.999), eps = 1e-8)
optimizerInc = optim.Adam(Inc.parameters(), lr = 0.00015, betas = (0.5, 0.999), eps = 1e-8)



dataset = datasets.ImageFolder(root='./dataset_2_dices', transform=transforms.Compose([
    # transforms.Resize(128),
    transforms.ToTensor(), 
]))

d2 = datasets.ImageFolder(root='./dataset_2_dices_validate', transform=transforms.Compose([
    # transforms.Resize(128),
    transforms.ToTensor(), 
]))
bs = 32
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)



try:
    os.mkdir('AE_2_dices_v2')
except OSError:
    pass

results_dir = './AE_2_dices_v2'

epochs = 140

for epoch in range(epochs+1):
    
        
    for idx, (images, _) in enumerate(dataloader):
        images = Variable(images).type('torch.cuda.FloatTensor').to(device)
        # images = images.to(device)
        generated = Encoder(images)
        generated = Dec(generated)
        generated = Inc(generated)
        recon_x = Decoder(generated)
        loss = F.mse_loss(recon_x, images, reduction='none').sum().mean()

        Encoder.zero_grad()
        Decoder.zero_grad()
        Dec.zero_grad()
        Inc.zero_grad()
        
        loss.backward()
        
        optimizerEncoder.step()
        optimizerDecoder.step()
        optimizerDec.step()
        optimizerInc.step()
        

    to_print = "Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.item()/bs)
    print(to_print)
    
    metrics['observation_loss'].append(loss.item()/bs)
    metrics['steps'].append(epoch)
    lineplot(metrics['steps'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
    
    if epoch % 10 == 0:
        fixed_x = d2[randint(1, 100)][0].unsqueeze(0).to(device)
        generated = Encoder(fixed_x)
        generated = Dec(generated)
        generated = Inc(generated)
        recon_x = Decoder(generated)
        
        compare_x = torch.cat([fixed_x, recon_x])
        
        save_image(compare_x.data.cpu(), './AE_2_dices_v2/image_%03d.png' % (epoch))   
        
    if epoch % 20 ==0:
        tempresults_dir = './AE_2_dices_v2/%s%d' % ('epoch',epoch)
        try:
            os.mkdir(tempresults_dir)
        except OSError:
            pass
        torch.save(Encoder,tempresults_dir+'/Encoder_%d.pt' % (epoch))
        torch.save(Decoder,tempresults_dir+'/Decoder_%d.pt' % (epoch))
        torch.save(Inc,tempresults_dir+'/Inc_%d.pt' % (epoch))
        torch.save(Dec,tempresults_dir+'/Dec_%d.pt' % (epoch))
        
