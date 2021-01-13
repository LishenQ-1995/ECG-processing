# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:57:17 2020

@author: Lishen Qiu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:28:45 2020

@author: Lishen Qiu
"""

#数据统一为 样本*长度 标签统一为 样本*2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:15:03 2020

@author: Lishen Qiu
"""
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

#from Disout import Disout,LinearScheduler
#from DANET import DAnet1,DAnet2,DAnet3

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

class conv_1_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch,kernel_size_L,kernel_size_W,stride):
        super(conv_1_block, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.ReLU(inplace=True),
            )

        self.ca =ChannelAttention(out_ch,8)
        self.sp =SpatialAttention(kernel_size_L,kernel_size_W,stride=1)
    def forward(self, x):
        
        x1 = self.conv1(x)
        xout = self.ca(x1)* x1

        
        return xout
    
class conv_2_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch,kernel_size_L,kernel_size_W,stride):
        super(conv_2_block, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.ReLU(inplace=True),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(kernel_size_L,kernel_size_W), stride=stride,padding=(0,kernel_size_W//2), bias=True),
            nn.ReLU(inplace=True),
            )
        self.ca =ChannelAttention(out_ch,8)
        self.sp =SpatialAttention(kernel_size_L,kernel_size_W,stride=1)
    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = self.ca(x1)* x1
        x1 = self.conv2(x1)
        x1 = self.ca(x1)* x1
        
#        print(x1.shape)
#        print(x.shape)
        xout=x1+x
        return xout
 

     
    
class DRnet(nn.Module):#库中的torch.nn.Module模块
    def __init__(self,in_channels =1):
        super(DRnet, self).__init__()
        
        
        self.conv1_1=conv_1_block( 2, 36, kernel_size_L=1,kernel_size_W=13, stride=1)
        self.conv1_2=conv_2_block(36, 36, kernel_size_L=1,kernel_size_W=13, stride=1)
        self.conv1_3=conv_2_block(36, 36, kernel_size_L=1,kernel_size_W=13, stride=1)
        self.conv1_4=conv_2_block(36, 36, kernel_size_L=1,kernel_size_W=13, stride=1)
        
        self.conv2_1=conv_1_block( 2, 36, kernel_size_L=1,kernel_size_W=15, stride=1)
        self.conv2_2=conv_2_block(36, 36, kernel_size_L=1,kernel_size_W=15, stride=1)
        self.conv2_3=conv_2_block(36, 36, kernel_size_L=1,kernel_size_W=15, stride=1)
        self.conv2_4=conv_2_block(36, 36, kernel_size_L=1,kernel_size_W=15, stride=1)
        
        self.conv3_1=conv_1_block( 2, 36, kernel_size_L=1,kernel_size_W=5, stride=1)
        self.conv3_2=conv_2_block(36, 36, kernel_size_L=1,kernel_size_W=5, stride=1)
        self.conv3_3=conv_2_block(36, 36, kernel_size_L=1,kernel_size_W=5, stride=1)
        self.conv3_4=conv_2_block(36, 36, kernel_size_L=1,kernel_size_W=5, stride=1)
        
        self.conv4_1=conv_1_block( 2, 36, kernel_size_L=1,kernel_size_W=3, stride=1)
        self.conv4_2=conv_2_block(36, 36, kernel_size_L=1,kernel_size_W=3, stride=1)
        self.conv4_3=conv_2_block(36, 36, kernel_size_L=1,kernel_size_W=3, stride=1)
        self.conv4_4=conv_2_block(36, 36, kernel_size_L=1,kernel_size_W=3, stride=1)
        
        self.conv1m1_1 = nn.Conv2d(in_channels=36, out_channels=1, kernel_size=(1,1),padding=0) 
        self.conv1m1_2 = nn.Conv2d(in_channels=36, out_channels=1, kernel_size=(1,1),padding=0) 
        self.conv1m1_3 = nn.Conv2d(in_channels=36, out_channels=1, kernel_size=(1,1),padding=0) 
        self.conv1m1_4 = nn.Conv2d(in_channels=36, out_channels=1, kernel_size=(1,1),padding=0) 
        
    def forward(self, x):

        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1m1_1(x1)
        
        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        x2 = self.conv2_3(x2)
        x2 = self.conv2_4(x2)
        x2 = self.conv1m1_2(x2)

        x3 = self.conv3_1(x)
        x3 = self.conv3_2(x3)
        x3 = self.conv3_3(x3)
        x3 = self.conv3_4(x3)
        x3 = self.conv1m1_3(x3)

        x4 = self.conv4_1(x)
        x4 = self.conv4_2(x4)
        x4 = self.conv4_3(x4)
        x4 = self.conv4_4(x4)
        x4 = self.conv1m1_4(x4)


        Xout=x1+x2+x3+x4

#        Xout = self.conv1m1(xout)

        return Xout
 
            
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = DRnet().to(device)
summary(model, (2,1,3600))#通道/长/宽



