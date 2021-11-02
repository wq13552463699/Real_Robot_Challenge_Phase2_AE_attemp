#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 00:37:30 2021

@author: qiang
"""
import torch
import torch.nn as nn

gpu_id = 0
device = torch.device("cuda", gpu_id)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=256):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, 4, 2, 1), # Output HxW = 64x64
            # nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1), # Output HxW = 32x32
            # nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), # Output HxW = 16x16
            # nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # Output HxW = 8x8
            # nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), # Output HxW = 4x4
            # nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 1024, 4, 2, 1), # Output HxW = 2x2
            nn.MaxPool2d((2,2)),
            Flatten()
        )
        
        # nn.Sequential(
        #     nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     Flatten()
        # )
        
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.Dropout(p=0.3),
            nn.ReLU(True),
            nn.Linear(512, 256),
            # nn.Dropout(p=0.3),
            nn.ReLU(True),
            )
        
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.Dropout(p=0.3),
            nn.ReLU(True),
            nn.Linear(512, 256),
            # nn.Dropout(p=0.3),
            nn.ReLU(True),
            )
        
        self.fc3 = nn.Sequential(
            nn.Linear(256, 512),
            # nn.Dropout(p=0.3),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            # nn.Dropout(p=0.3),
            nn.ReLU(True),
            )
        
        self.decoder = nn.Sequential(
            
            UnFlatten(),
            nn.ConvTranspose2d(1024, 256, 4, 1, 0, bias = False), # Output HxW = 4x4
            # nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), # Output HxW = 8x8
            # nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), # Output HxW = 16x16
            # nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False), # Output HxW = 32x32
            # nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias = False), # Output HxW = 64x64
            # nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias = False), # Output HxW = 128x128
            nn.Tanh()
        )
        
        # nn.Sequential(
            
        #     nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
        #     nn.Sigmoid(),
        # )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
