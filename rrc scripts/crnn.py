# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 05:31:02 2021

@author: 14488
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable

import torch
import rrc_example_package.scripts.convolutional_rnn as convolutional_rnn
from torch.nn.utils.rnn import pack_padded_sequence

asize = 1

def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


''' Generator network for 128x128 RGB images '''
class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.main = nn.Sequential(
            # Input HxW = 128x128
            nn.Conv2d(3, 16, 4, 2, 0), # Output HxW = 134
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 0), # Output HxW = 66
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 0), # Output HxW = 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # Output HxW = 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # Output HxW = 8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), # Output HxW = 4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 4, 2, 1), # Output HxW = 2
            nn.MaxPool2d((2,2)),
            # At this point, we arrive at our low D representation vector, which is 512 dimensional.
        )

    
    def forward(self, input):
        output = self.main(input)
        return output



class Decoder(nn.Module):
    
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.main = nn.Sequential(

            nn.ConvTranspose2d(1024,512, 4, 1, 0, bias = False), # Output HxW = 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), # Output HxW = 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), # Output HxW = 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), # Output HxW = 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 0, bias = False), # Output HxW = 66
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 0, bias = False), # Output HxW = 134
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 0, bias = False), # Output HxW = 270
            nn.Tanh()
        )

    
    def forward(self, input):
        output = self.main(input)
        return output

class RNN(nn.Module):
    
    def __init__(self):
        super(RNN, self).__init__()
        
        self.main = nn.GRU(1024, 1024, 1)
        
    def forward(self, input,hx):
        input = input.view(1,1, 1024)
        output,hn = self.main(input,hx)
        output = output.view(1, 1024, 1, 1)
        return output,hn

''' Discriminator network for 128x128 RGB images '''

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.main = convolutional_rnn.Conv2dGRU(in_channels=3,  # Corresponds to input size
                                    out_channels=3,  # Corresponds to hidden size
                                    kernel_size=4,  # Int or List[int]
                                    num_layers=2,
                                    bidirectional=True,
                                    dilation=2, stride=1, dropout=0.5,
                                    batch_first=True)

    def forward(self, input,hx):
        input = input.view(1,1,3,270,270)
        output,hn = self.main(input,hx)
        output = output.view(1,6,270,270)
        return output,hn
    
class Dis(nn.Module):
    
    def __init__(self):
        super(Dis, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid())
    
    
    def forward(self, input):
        output = self.main(input)
        return output.view(-1)
    
    
class Dec(nn.Module):
    
    def __init__(self):
        super(Dec, self).__init__()
        
        self.main = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU())

    
    def forward(self, input):
        input = input.view(-1, 1024)
        output = self.main(input)
        return output

class Inc(nn.Module):
    
    def __init__(self):
        super(Inc, self).__init__()
        
        self.main = self.main = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU())

    
    def forward(self, input):
        output = self.main(input)
        output = output.view(1, 1024, 1, 1)
        return output
    
class Dec2(nn.Module):
    
    def __init__(self):
        super(Dec2, self).__init__()
        
        self.main = nn.Linear(256, 128)

    
    def forward(self, input):
        output = self.main(input)
        return output

class Inc2(nn.Module):
    
    def __init__(self):
        super(Inc2, self).__init__()
        
        self.main = nn.Linear(128, 256)

    
    def forward(self, input):
        output = self.main(input)
        return output