# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:15:03 2020

@author: Lishen Qiu
"""
from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import torch
from numpy.linalg import svd
from numpy.random import normal
import math
from math import sqrt
from torchsummary import summary
import scipy.io as io
import numpy as np
import torch.optim as optim
import torch.utils.data 
import torch
import os
import os.path as osp
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader



#class SELayer(nn.Module):
#    def __init__(self, channel, reduction):
#        super(SELayer, self).__init__()
#        self.avg_pool = nn.AdaptiveAvgPool2d(1)
##        print(self.avg_pool.size())
#        self.fc = nn.Sequential(
#            nn.Linear(channel, channel // reduction, bias=False),
#            nn.ReLU(inplace=True),
#            nn.Linear(channel // reduction, channel, bias=False),
#            nn.Sigmoid()
#        )
#
#    def forward(self, x):
#        b, c, _, _ = x.size()
#        y = self.avg_pool(x).view(b, c)
#        y = self.fc(y).view(b, c, 1, 1)
#        return x * y.expand_as(x)

    

class conv_3_block_DW(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch,kernel_size_L,kernel_size_W,stride):
        super(conv_3_block_DW, self).__init__()
        
        self.conv = nn.Sequential(
                
           
            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
#            SELayer(out_ch, 8),
            nn.Conv2d(out_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
#            SELayer(out_ch, 8),
            nn.Conv2d(out_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
#            SELayer(out_ch, 8),
            )
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class conv_3_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch,kernel_size_L,kernel_size_W,stride):
        super(conv_3_block, self).__init__()
        
        self.conv = nn.Sequential(
                
           
            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
#            SELayer(out_ch, 8),
            nn.Conv2d(out_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
#            SELayer(out_ch, 8),
            nn.Conv2d(out_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
#            SELayer(out_ch, 8),
            )
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class conv_3_block_UP(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch,kernel_size_L,kernel_size_W,stride):
        super(conv_3_block_UP, self).__init__()
        
        self.conv = nn.Sequential(
                
           
            nn.Conv2d(in_ch, in_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
#            SELayer(in_ch, 8),
            nn.Conv2d(in_ch, in_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
#            SELayer(in_ch, 8),
            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
#            SELayer(out_ch, 8),
            )
        
    def forward(self, x):
        x = self.conv(x)
        return x  
       
class FCN(nn.Module):#库中的torch.nn.Module模块
    def __init__(self,in_channels =1):
        super(FCN, self).__init__()
        
        self.conv1=conv_3_block_DW( 1, 16, kernel_size_L=1,kernel_size_W=25,stride=1)
        self.conv2=conv_3_block_DW(16, 32, kernel_size_L=1,kernel_size_W=15,stride=1)
        self.conv3=conv_3_block_DW(32, 48, kernel_size_L=1,kernel_size_W=5,stride=1)
        
        self.conv4=conv_3_block(48, 48, kernel_size_L=1,kernel_size_W=3,stride=1)
        
        self.conv5=conv_3_block_UP(48, 32, kernel_size_L=1,kernel_size_W=5,stride=1)
        self.conv6=conv_3_block_UP(32, 16, kernel_size_L=1,kernel_size_W=15,stride=1)
        self.conv7=conv_3_block_UP(16, 16, kernel_size_L=1,kernel_size_W=25,stride=1)
        
        self.conv1m1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1,1),padding=0) 
        
        self.avepool1 = nn.AvgPool2d((1, 5), stride=5) 
        self.avepool2 = nn.AvgPool2d((1, 2), stride=2)
        self.avepool3 = nn.AvgPool2d((1, 2), stride=2)
              
        self.up1 = nn.Upsample(size=(1, 360), scale_factor=None, mode='bilinear', align_corners=None)  
        self.up2 = nn.Upsample(size=(1, 720), scale_factor=None, mode='bilinear', align_corners=None)  
        self.up3 = nn.Upsample(size=(1, 3600), scale_factor=None, mode='bilinear', align_corners=None)  
    
        
    def forward(self, x):
        
        x1_1 = self.conv1(x)
        x1_2 = self.avepool1(x1_1)
        x1_3 = self.conv2(x1_2)
        x1_4 = self.avepool2(x1_3)
        x1_5 = self.conv3(x1_4)
        x1_6 = self.avepool3(x1_5)
        
        x2 = self.conv4(x1_6)
        
        x2_1 = self.up1(x2)
#        x2_1 = torch.cat((x2_1, x1_5), 1)  # x7 cat x3
        x2_2 = self.conv5(x2_1)
        x2_3 = self.up2(x2_2)
#        x2_3 = torch.cat((x2_3, x1_3), 1)  # x7 cat x3
        x2_4 = self.conv6(x2_3)
        x2_5 = self.up3(x2_4)
#        x2_5 = torch.cat((x2_5, x1_1), 1)  # x7 cat x3
        x2_6 = self.conv7(x2_5)
        Xout = self.conv1m1(x2_6)

        return Xout
 
            
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = FCN().to(device)
summary(model, (1,1,3600))