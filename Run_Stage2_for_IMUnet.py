# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:28:45 2020

@author: Lishen Qiu
"""

import torch.nn as nn
import torch
#import torchvision
import torch.utils.data 
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
import numpy as np
import torch.optim as optim
from torchsummary import summary
import time
from torch.autograd import Variable
from numpy.random import normal
from math import sqrt
import math
import torch.nn.functional as F
#from loss2 import generalized_dice_loss,dice_loss
from Stage2_model3 import DRnet
import os
#from circle_loss import convert_label_to_similarity, CircleLoss

#数据统一为 样本*长度 标签统一为 样本*2
def Datacombination(data1, data2,data3):
    train_1=torch.cat((data1,data2,data3), dim=2) 
    train_2=torch.cat((data2,data1,data3), dim=2) 
    train_3=torch.cat((data1,data3,data2), dim=2) 
    train_all= torch.cat((train_1,train_2,train_3), dim=1) 
    return train_all

def SNR(y_true, y_pred):
    temp1=torch.sum(torch.pow(y_true,2),dim=(1,2,3))
    temp2=torch.sum(torch.pow(y_true-y_pred,2),dim=(1,2,3))
    snr=10*torch.log(temp1/temp2)/np.log(10.0)#torch.log是以自然数e为底的对数函数
    return snr

sigwithnoise_test = io.loadmat('/home/ps/QLS/双阶段去噪实验/0.6bw/sigwithnoise_test.mat')
sigwithnoise_test = sigwithnoise_test['sigwithnoise_test']#3000*4000

sig_all_test = io.loadmat('/home/ps/QLS/双阶段去噪实验/0.6bw/sig_all_test.mat')
sig_all_test = sig_all_test['sig_all_test']#3000*4000

sigwithnoise_validation = io.loadmat('/home/ps/QLS/双阶段去噪实验/0.6bw/sigwithnoise_validation.mat')
sigwithnoise_validation = sigwithnoise_validation['sigwithnoise_validation']#3000*4000

sig_all_validation = io.loadmat('/home/ps/QLS/双阶段去噪实验/0.6bw/sig_all_validation.mat')
sig_all_validation = sig_all_validation['sig_all_validation']#3000*4000

sigwithnoise_train = io.loadmat('/home/ps/QLS/双阶段去噪实验/0.6bw/sigwithnoise_train.mat')
sigwithnoise_train = sigwithnoise_train['sigwithnoise_train']#3000*4000

sig_all_train = io.loadmat('/home/ps/QLS/双阶段去噪实验/0.6bw/sig_all_train.mat')
sig_all_train = sig_all_train['sig_all_train']#3000*4000


print(sigwithnoise_test.shape)
print(sig_all_test.shape)
print(sigwithnoise_validation.shape)
print(sig_all_validation.shape)
print(sigwithnoise_train.shape)
print(sig_all_train.shape)


sigwithnoise_test = torch.Tensor(sigwithnoise_test)
sig_all_test = torch.Tensor(sig_all_test)#([1500, 1, 4000])
sigwithnoise_validation = torch.Tensor(sigwithnoise_validation)
sig_all_validation = torch.Tensor(sig_all_validation)#([1500, 1, 4000])
sigwithnoise_train = torch.Tensor(sigwithnoise_train)#([30732, 2, 2000])
sig_all_train = torch.Tensor(sig_all_train)

sigwithnoise_test = torch.unsqueeze(sigwithnoise_test, 1)
sigwithnoise_test = torch.unsqueeze(sigwithnoise_test, 1)
sig_all_test = torch.unsqueeze(sig_all_test, 1)
sig_all_test = torch.unsqueeze(sig_all_test, 1)
sigwithnoise_validation = torch.unsqueeze(sigwithnoise_validation, 1)
sigwithnoise_validation = torch.unsqueeze(sigwithnoise_validation, 1)
sig_all_validation = torch.unsqueeze(sig_all_validation, 1)
sig_all_validation = torch.unsqueeze(sig_all_validation, 1)
sigwithnoise_train = torch.unsqueeze(sigwithnoise_train, 1)
sigwithnoise_train = torch.unsqueeze(sigwithnoise_train, 1)
sig_all_train = torch.unsqueeze(sig_all_train, 1)
sig_all_train = torch.unsqueeze(sig_all_train, 1)

pred_train = io.loadmat('/home/ps/QLS/双阶段去噪实验/Stage_1/IMUnet第一阶段的结果/pred_train.mat')
pred_train = pred_train['pred_train']#3000*4000

pred_vaild = io.loadmat('/home/ps/QLS/双阶段去噪实验/Stage_1/IMUnet第一阶段的结果/pred_vaild.mat')
pred_vaild = pred_vaild['pred_vaild']#3000*4000

pred_test = io.loadmat('/home/ps/QLS/双阶段去噪实验/Stage_1/IMUnet第一阶段的结果/pred_test.mat')
pred_test = pred_test['pred_test']#3000*4000

pred_train = torch.Tensor(pred_train)
pred_vaild = torch.Tensor(pred_vaild)
pred_test = torch.Tensor(pred_test)

train_all= torch.cat((sigwithnoise_train,pred_train), dim=1) 
vaild_all= torch.cat((sigwithnoise_validation,pred_vaild), dim=1) 
test_all = torch.cat((sigwithnoise_test,pred_test), dim=1) 

print(train_all.shape)#torch.Size([965, 3, 3, 3600])
print(vaild_all.shape)#torch.Size([276, 3, 3, 3600])
print(test_all.shape)#torch.Size([138, 3, 3, 3600])

"""
数据进入dataset
"""
train_dataset = torch.utils.data.TensorDataset(train_all, sig_all_train)
vaild_dataset = torch.utils.data.TensorDataset(vaild_all, sig_all_validation)
test_dataset = torch.utils.data.TensorDataset(test_all, sig_all_test)

train_data_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)#加了 num_workers=2 更慢
vaild_data_loader = DataLoader(dataset=vaild_dataset, batch_size=64, shuffle=False)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

"""
训练的准备
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = DRnet().to(device)

#metricsfun=nn.MSELoss(reduction='mean')#output = loss(input, target)
#criterion = nn.MSELoss(reduction='mean')

#criterion = nn.MSELoss(reduction='mean')
criterion = nn.L1Loss(reduction='mean')
metricsfun= SNR

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
global best_loss
best_loss = float('inf')  
global best_epoch
best_epoch=0
global best_SNR
best_SNR=0

train_loss_record = []
val_loss_record = []
test_loss_record = []

train_SNR_record = []
val_SNR_record = []
test_SNR_record = []

train_SNR_all = []
val_SNR_all = []
test_SNR_all = []

# Training
def train(epoch):
    print("=======================")
    t3= time.time()

    model.train()
    train_loss = 0
    SNR_train=0
    for batch_idx,(input,output) in enumerate(train_data_loader):#input,output 为每一个batch的输入输出 大小也是batch

        input = input.cuda()#cuda型
        output = output.cuda()#cuda型

        input = Variable(input)#(variable)变量是可以只用GPU进行加速计算的
        output = Variable(output)#(variable)变量是可以只用GPU进行加速计算的
#
        optimizer.zero_grad()#清零一下 当网络参量进行反馈时，梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，
        preds = model(input)
        
        loss = criterion(preds,output)
        SNRnum= metricsfun(output,preds)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()#item()方法 是得到一个元素张量里面的元素值

        SNR_train = SNRnum.data.cpu().numpy()
        SNR_train = list(SNR_train)
        train_SNR_all.extend(SNR_train)

        if batch_idx % 5==0:
            print("Trian---Epoch=%d, batch_idx=%d, this_batch_loss=%f"%(epoch,batch_idx,loss))
            
    train_loss=train_loss/len(train_dataset)
    train_loss_record.append(train_loss)    
    
    train_all_ave=sum(train_SNR_all)/len(train_SNR_all)
    train_SNR_record.append(train_all_ave)  

    torch.cuda.synchronize()#用在time上
    
    t4 = time.time()
    timeSpan=t4-t3
    print("------------------")
    print("------------------")
    print("Trian---Epoch=%d, train_loss=%f*10-3, train_SNR=%f, 耗时%.3fS"%(epoch,train_loss*1000,train_all_ave,timeSpan))
    print("------------------")
    
def val(epoch):
    
    model.eval()
    val_loss = 0
    SNR_val=0
    
    for batch_idx,(input,output) in enumerate(vaild_data_loader):
        input = input.cuda()
        output = output.cuda()
        input = Variable(input)
        output = Variable(output)
#        print(batch_idx)
        preds = model(input)
#        loss = criterion(output,preds)
        loss = criterion(preds, output)
        SNRnum= metricsfun(output,preds)
        val_loss += loss.data.item()
        
        SNR_val = SNRnum.data.cpu().numpy()#用list前的操作
        SNR_val = list(SNR_val)
        val_SNR_all.extend(SNR_val)


    global best_loss
    global best_epoch
    global best_SNR
    val_loss=val_loss/len(vaild_dataset)
    val_loss_record.append(val_loss) 
    
    
    val_all_ave=sum(val_SNR_all)/len(val_SNR_all)
    val_SNR_record.append(val_all_ave) 
    

    print('val---val_loss=%.6f*10-3, val_SNR=%.6f' % (val_loss*1000, val_all_ave))
    if val_loss < best_loss:

        torch.save(model.state_dict(), '/home/ps/QLS/双阶段去噪实验/Stage_2/第二阶段_IMUnet的结果/IMUnet_DRnet.pth')
        best_epoch=epoch
        best_loss = val_loss
        best_SNR=val_all_ave

        print(' Model Saved ! best_epoch=%d, val_best_loss=%.6f*10-3, val_best_SNR=%.6f'% (best_epoch, best_loss*1000,best_SNR))
        print("------------------")
        
    else:
        print('Pass || best_epoch=%d, val_best_loss=%.6f*10-3, val_best_SNR=%.6f'% (best_epoch, best_loss*1000,best_SNR))
        print("------------------")

        

def test(epoch):
    
    model.eval()
    SNR_test=0
    test_loss = 0
    for batch_idx,(input,output) in enumerate(test_data_loader):
        input = input.cuda()
        output = output.cuda()
        input = Variable(input)
        output = Variable(output)
#        print(batch_idx)
        preds = model(input)
#        loss = criterion(output,preds)
        loss = criterion(preds, output)
        SNRnum= metricsfun(output,preds)
        
        test_loss += loss.data.item()
        
        SNR_test = SNRnum.data.cpu().numpy()
        SNR_test = list(SNR_test)
        test_SNR_all.extend(SNR_test)
        
        
    test_loss=test_loss/len(test_dataset)
    test_loss_record.append(test_loss) 
    
    test_all_ave=sum(test_SNR_all)/len(test_SNR_all)
    test_SNR_record.append(test_all_ave)  
    
#    SNR_test=SNR_test/len(test_dataset)
#    test_SNR_record.append(SNR_test) 
    
    print('Test---test_loss=%.6f*10-3, test_SNR=%.6f' % (test_loss*1000, test_all_ave))
    print("=======================")
    print('*')

start_epoch = 0
for epoch in range(start_epoch, start_epoch+1000):
    train(epoch)
    val(epoch)
    test(epoch)
    
io.savemat('/home/ps/QLS/双阶段去噪实验/Stage_2/第二阶段_IMUnet的结果/train_loss_record.mat',{'train_loss_record':train_loss_record})
io.savemat('/home/ps/QLS/双阶段去噪实验/Stage_2/第二阶段_IMUnet的结果/val_loss_record.mat',{'val_loss_record':val_loss_record})
io.savemat('/home/ps/QLS/双阶段去噪实验/Stage_2/第二阶段_IMUnet的结果/test_loss_record.mat',{'test_loss_record':test_loss_record})

io.savemat('/home/ps/QLS/双阶段去噪实验/Stage_2/第二阶段_IMUnet的结果/train_SNR_record.mat',{'train_SNR_record':train_SNR_record})
io.savemat('/home/ps/QLS/双阶段去噪实验/Stage_2/第二阶段_IMUnet的结果/val_SNR_record.mat',{'val_SNR_record':val_SNR_record})
io.savemat('/home/ps/QLS/双阶段去噪实验/Stage_2/第二阶段_IMUnet的结果/test_SNR_record.mat',{'test_SNR_record':test_SNR_record})

def pred_sig(batch_size,sig_07):
    sig=sig_07
    batch_size=batch_size
    
    T=math.ceil(len(sig)/batch_size)
    #T=math.ceil(16638/batch_size)
    
    T1=list(range(0,T,1))
    pred_all= torch.zeros(1,1,1,3600).to(device)
    #pred_all=torch.pred_all
    for i in T1:
        print(i)
        with torch.no_grad():
            
            pred_batch = model(sig[(i)*batch_size:(i+1)*batch_size]) 
#           pred_07 = model(sig_07[0:128])
#           print(pred_batch.shape)
            pred_all=torch.cat((pred_all,pred_batch), dim=0) 
    pred_all=pred_all.data.cpu().numpy()
    pred_all=pred_all[1:]
    print('finished')
    return pred_all

model.load_state_dict(torch.load('/home/ps/QLS/双阶段去噪实验/Stage_2/第二阶段_IMUnet的结果/IMUnet_DRnet.pth'))#读模型
model.eval()#进入评估模式

train_all = train_all.to(device)
pred_train=pred_sig(64,train_all)
io.savemat('/home/ps/QLS/双阶段去噪实验/Stage_2/第二阶段_IMUnet的结果/pred_train.mat',{'pred_train':pred_train})

vaild_all = vaild_all.to(device)
pred_vaild=pred_sig(64,vaild_all)
io.savemat('/home/ps/QLS/双阶段去噪实验/Stage_2/第二阶段_IMUnet的结果/pred_vaild.mat',{'pred_vaild':pred_vaild})

test_all = test_all.to(device)
pred_test=pred_sig(64,test_all)
io.savemat('/home/ps/QLS/双阶段去噪实验/Stage_2/第二阶段_IMUnet的结果/pred_test.mat',{'pred_test':pred_test})

