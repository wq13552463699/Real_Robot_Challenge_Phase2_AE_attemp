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
from torchvision.utils import save_image
from rrc_example_package.scripts.utils import lineplot
from torchvision import datasets
import cv2
from rrc_example_package import rearrange_dice_env_GAN
from copy import copy
from rrc_example_package.scripts.vae_network import VAE
from torch.autograd import Variable
from torchvision import transforms
from random import randint

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

model = VAE(image_channels=3).to(device)
# model.load_state_dict(torch.load('vae.torch', map_location='cpu'))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


dataset = datasets.ImageFolder(root='./dataset', transform=transforms.Compose([
    # transforms.Resize(128),
    transforms.ToTensor(), 
]))

d2 = datasets.ImageFolder(root='./validate', transform=transforms.Compose([
    # transforms.Resize(128),
    transforms.ToTensor(), 
]))

bs = 30
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)


def compare(x):
    recon_x, mu, logvar = model(x)
    print(loss_fn(recon_x, x, mu, logvar))
    return torch.cat([x, recon_x])

try:
    os.mkdir('VAE_1')
except OSError:
    pass

results_dir = './VAE_1'

epochs = 500
for epoch in range(epochs):
    
        
    for idx, (images, _) in enumerate(dataloader):
        images = images.to(device)
        # print(images.shape)
        recon_images, mu, logvar = model(images)
        loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
        # print(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()

    to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                            epochs, loss.item()/bs, bce.item()/bs, kld.item()/bs)
    print(to_print)
    
    if epoch % 10 == 0:
        fixed_x = d2[randint(1, 70)][0].unsqueeze(0).to(device)
        compare_x = compare(fixed_x)
        save_image(compare_x.data.cpu(), './VAE_1/image_%03d.png' % (epoch))

 
    if epochs % 100 ==0:
        torch.save(model.state_dict(), './VAE_1/vae.torch')







