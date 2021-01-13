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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size_L,kernel_size_W,stride):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
#        print(self.avg_pool.size())
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    

class conv_1_block_DW(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch,kernel_size_L,kernel_size_W,stride):
        super(conv_1_block_DW, self).__init__()
        
        self.conv = nn.Sequential(
                
           
            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
#            SELayer(out_ch, 8),

            )
        self.ca =ChannelAttention(out_ch,8)
        self.sp =SpatialAttention(kernel_size_L,kernel_size_W,stride=1)
    def forward(self, x):
        x = self.conv(x)
        x = self.ca(x)* x
#        x = self.sp(x)* x
        return x
    
class conv_1_block_MD(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch,kernel_size_L,kernel_size_W,stride):
        super(conv_1_block_MD, self).__init__()
        
        self.conv = nn.Sequential(
                
           
            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
#            SELayer(out_ch, 8),

            )
        self.ca =ChannelAttention(out_ch,8)
        self.sp =SpatialAttention(kernel_size_L,kernel_size_W,stride=1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.ca(x)* x
#        x = self.sp(x)* x
        return x

class conv_1_block_UP(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch,kernel_size_L,kernel_size_W,stride):
        super(conv_1_block_UP, self).__init__()
        
        self.conv = nn.Sequential(
                
           
            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
#            SELayer(out_ch, 8),
            )
        self.ca =ChannelAttention(out_ch,8)
        self.sp =SpatialAttention(kernel_size_L,kernel_size_W,stride=1)
    def forward(self, x):
        x = self.conv(x)
        x = self.ca(x)* x
#        x = self.sp(x)* x
        return x  
       
class Context_comparison(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch,kernel_size_L,kernel_size_W,stride):
        super(Context_comparison, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,(kernel_size_W+7)//2), bias=True,dilation=5),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            )
        self.ca =ChannelAttention(out_ch,8)
        self.sp =SpatialAttention(kernel_size_L,kernel_size_W,stride=1)
        self.conv1m1 = nn.Conv2d(in_channels=out_ch*3, out_channels=out_ch, kernel_size=(1,1),padding=0) 
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.ca(x1)* x1
        x1 = self.conv1(x1)
        x1 = self.ca(x1)* x1
        
        x2 = self.conv2(x)
        x2 = self.ca(x2)* x2
        x2 = self.conv2(x2)
        x2 = self.ca(x2)* x2
#        print(x1.shape)
#        print(x2.shape)
        x3=x1-x2
#        print(x1.shape)
#        print(x3.shape)
        xout = torch.cat((x1,x2,x3), 1)  
#        print(xout.shape)
        xout=self.conv1m1(xout)
        return xout          
    
class IMUnet(nn.Module):#库中的torch.nn.Module模块
    def __init__(self,in_channels =1):
        super(IMUnet, self).__init__()
        
        self.conv1_1=conv_1_block_DW( 1, 16, kernel_size_L=1,kernel_size_W=25,stride=1)
        self.conv1_2=conv_1_block_DW(16, 16, kernel_size_L=1,kernel_size_W=25,stride=1)
        self.conv1_3=conv_1_block_DW(16, 16, kernel_size_L=1,kernel_size_W=25,stride=1)
        
        self.conv2_1=conv_1_block_DW(16, 32, kernel_size_L=1,kernel_size_W=15,stride=1)
        self.conv2_2=conv_1_block_DW(32, 32, kernel_size_L=1,kernel_size_W=15,stride=1)
        self.conv2_3=conv_1_block_DW(32, 32, kernel_size_L=1,kernel_size_W=15,stride=1)
        
        self.conv3_1=conv_1_block_DW(32, 48, kernel_size_L=1,kernel_size_W=5,stride=1)
        self.conv3_2=conv_1_block_DW(48, 48, kernel_size_L=1,kernel_size_W=5,stride=1)
        self.conv3_3=conv_1_block_DW(48, 48, kernel_size_L=1,kernel_size_W=5,stride=1)
        
        self.conv4_1=conv_1_block_MD(48, 64, kernel_size_L=1,kernel_size_W=3,stride=1)
#        self.conv4_2=conv_1_block_MD(64, 64, kernel_size_L=1,kernel_size_W=3,stride=1)
        self.conv4_2=Context_comparison(64, 64, kernel_size_L=1,kernel_size_W=3,stride=1)
        self.conv4_3=conv_1_block_MD(64, 64, kernel_size_L=1,kernel_size_W=3,stride=1)
        
        self.conv5_1=conv_1_block_UP(48+64, 48, kernel_size_L=1,kernel_size_W=5,stride=1)
        self.conv5_2=conv_1_block_UP(48, 48, kernel_size_L=1,kernel_size_W=5,stride=1)
        self.conv5_3=conv_1_block_UP(48, 32, kernel_size_L=1,kernel_size_W=5,stride=1)
        
        self.conv6_1=conv_1_block_UP(32+32, 32, kernel_size_L=1,kernel_size_W=15,stride=1)
        self.conv6_2=conv_1_block_UP(32, 32, kernel_size_L=1,kernel_size_W=15,stride=1)
        self.conv6_3=conv_1_block_UP(32, 16, kernel_size_L=1,kernel_size_W=15,stride=1)
        
        self.conv7_1=conv_1_block_UP(16+16, 16, kernel_size_L=1,kernel_size_W=25,stride=1)
        self.conv7_2=conv_1_block_UP(16, 16, kernel_size_L=1,kernel_size_W=25,stride=1)
        self.conv7_3=conv_1_block_UP(16, 16, kernel_size_L=1,kernel_size_W=25,stride=1)
        
        self.conv1m1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1,1),padding=0) 
        
        self.avepool1 = nn.AvgPool2d((1, 5), stride=5) 
        self.avepool2 = nn.AvgPool2d((1, 2), stride=2)
        self.avepool3 = nn.AvgPool2d((1, 2), stride=2)
              
        self.up1 = nn.Upsample(size=(1, 360), scale_factor=None, mode='bilinear', align_corners=None)  
        self.up2 = nn.Upsample(size=(1, 720), scale_factor=None, mode='bilinear', align_corners=None)  
        self.up3 = nn.Upsample(size=(1, 3600), scale_factor=None, mode='bilinear', align_corners=None)  
        

    def forward(self, x):# print(x.shape)

        x1_1 = self.conv1_1(x)
        x1_2 = self.conv1_2(x1_1)
        x1_3 = self.conv1_3(x1_2)        
        x1 = self.avepool1(x1_3)

        x2_1 = self.conv2_1(x1)
        x2_2 = self.conv2_2(x2_1)
        x2_3 = self.conv2_3(x2_2)
        x2 = self.avepool2(x2_3)

        x3_1 = self.conv3_1(x2)
        x3_2 = self.conv3_2(x3_1)
        x3_3 = self.conv3_3(x3_2)
        x3 = self.avepool3(x3_3)
        
        x4_1 = self.conv4_1(x3)
        x4_2 = self.conv4_2(x4_1)
        x4  = self.conv4_3(x4_2)

        x4 = self.up1(x4)
#        print(x4.shape)
#        print(x3_3.shape)
        x4 = torch.cat((x4, x3_3), 1)  
        x5_1 = self.conv5_1(x4)
        x5_1=x5_1.add(x3_3)
#        x5_1 = torch.add((x5_1, x3_3))  
        x5_2 = self.conv5_2(x5_1)
#        x5_2 = torch.add((x5_2, x3_3))  
        x5_2=x5_2.add(x3_3)
        x5 = self.conv5_3(x5_2)

        x5 = self.up2(x5)
        x5 = torch.cat((x5, x2_3), 1)  
        x6_1 = self.conv6_1(x5)
#        x6_1 = torch.add((x6_1, x2_3))  
        x6_1=x6_1.add(x2_3)
        x6_2 = self.conv6_2(x6_1)  
        x6_2=x6_2.add(x2_3)
        x6 = self.conv6_3(x6_2)


        x6 = self.up3(x6)
        x6 = torch.cat((x6, x1_3), 1)  
        x7_1 = self.conv7_1(x6)
        x7_1=x7_1.add(x1_3)
        x7_2 = self.conv7_2(x7_1)
        x7_2=x7_2.add(x1_3)
        x7 = self.conv7_3(x7_2)

        Xout = self.conv1m1(x7)

        return Xout
 
            
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = IMUnet().to(device)
summary(model, (1,1,3600))