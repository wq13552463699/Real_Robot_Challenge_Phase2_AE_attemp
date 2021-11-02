#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 01:27:59 2021

@author: qiang
"""

import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class Encoder(nn.Module):
    def __init__(self, image_channels=3):
        super(Encoder, self).__init__()
        
        self.main = nn.Sequential(
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
        
    def forward(self, input):
        output = self.main(input)
        return output

class Decoder(nn.Module):
    
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.main = nn.Sequential(
            
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
        
    def forward(self, input):
        output = self.main(input)
        return output
        
class Dec(nn.Module):
    
    def __init__(self):
        super(Dec, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(1024, 16),
            # nn.Dropout(p=0.3),
            nn.ReLU(True),
            )

    
    def forward(self, input):
        output = self.main(input)
        return output

class Inc(nn.Module):
    
    def __init__(self):
        super(Inc, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(16, 1024),
            # nn.Dropout(p=0.3),
            nn.ReLU(True),
            )

    
    def forward(self, input):
        output = self.main(input)
        return output
        
    # def reparameterize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     # return torch.normal(mu, std)
    #     esp = torch.randn(*mu.size()).to(device)
    #     z = mu + std * esp
    #     return z
    
    # def bottleneck(self, h):
    #     mu, logvar = self.fc1(h), self.fc2(h)
    #     z = self.reparameterize(mu, logvar)
    #     return z, mu, logvar

    # def encode(self, x):
    #     h = self.encoder(x)
    #     z, mu, logvar = self.bottleneck(h)
    #     return z, mu, logvar

    # def decode(self, z):
    #     z = self.fc3(z)
    #     z = self.decoder(z)
    #     return z

    # def forward(self, x):
    #     z, mu, logvar = self.encode(x)
    #     z = self.decode(z)
    #     return z, mu, logvar
