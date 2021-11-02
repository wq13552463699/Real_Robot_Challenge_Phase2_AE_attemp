#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 20:23:44 2021

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
        return input.view(1, size, 1, 1)


class Coor(nn.Module):
    
    def __init__(self):
        super(Coor, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(384, 75),
            nn.ReLU(True),
            # nn.Linear(512, 256),
            # nn.ReLU(True),
            # nn.Linear(256, 128),
            # nn.ReLU(True),
            # nn.Linear(128, 75),
            # nn.ReLU(True),
            nn.Linear(75, 75),
            nn.ReLU(True),
            )

    
    def forward(self, input):
        output = self.main(input)
        return output

# class Decoder(nn.Module):
    
#     def __init__(self):
#         super(Decoder, self).__init__()
        
#         self.main = nn.Sequential(
            
#             nn.Linear(75, 1024),
#             nn.ReLU(True),
            
#             UnFlatten(),
            
#             nn.ConvTranspose2d(1024, 256, 4, 2, 0, bias = False), # Output HxW = 4x4
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), # Output HxW = 8x8
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), # Output HxW = 16x16
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False), # Output HxW = 32
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 16, 4, 2, 0, bias = False), # Output HxW = 66
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 4, 2, 0, bias = False), # Output HxW = 134
#             nn.BatchNorm2d(8),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 3, 4, 2, 0, bias = False), # Output HxW = 270
            
#             nn.Tanh()
#         )
        
#     def forward(self, input):
#         output = self.main(input)
#         return output

class Decoder(nn.Module):
    
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.main = nn.Sequential(
            
            nn.Linear(75, 1024),
            nn.ReLU(True),
            
            UnFlatten(),
            
            nn.ConvTranspose2d(1024, 256, 4, 2, 0, bias = False), # Output HxW = 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), # Output HxW = 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), # Output HxW = 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False), # Output HxW = 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias = False), # Output HxW = 66
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias = False), # Output HxW = 128
            # nn.BatchNorm2d(8),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(8, 3, 4, 2, 0, bias = False), # Output HxW = 270
            
            nn.Tanh()
        )
        
    def forward(self, input):
        output = self.main(input)
        return output