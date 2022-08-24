import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import datetime
import os
from prefetch_generator import BackgroundGenerator
# import lmdb
import pickle
import gc 

# dataloader
def sigmoid_unettest(x,center = 0.999, scale=5):
    u = (x-center)/(1-center)*scale
    return 1/(1+np.exp(-u))

def dchis_cal(lc,binary,single,err,weight=1):
    lc = lc.astype(np.float64)
    binary = binary.astype(np.float64)
    single = single.astype(np.float64)
    err = err.astype(np.float64)
    return np.sum(((lc-single)**2/err**2-(lc-binary)**2/err**2)*weight)


def default_loader(data_root,posi_lc,num_skip,residual=False):
    ## argsdata: [u_0, rho, q, s, alpha, t_E, basis_m, t_0]
    ## args_singlefitting: [t_E,t_0,u_0,basis_m]
    ## [args_data, arg_singlefitting,time,d_time,lc_withnoi,err,lc_withoutnoi,lc_singlemodel,unet_label]

    datadir = list(np.load(data_root+str(posi_lc+num_skip)+".npy", allow_pickle=True))

    extra_noise_index = np.array(datadir[9],dtype=np.int)
    extra_noise = np.array(datadir[10],dtype=np.float64)

    lc_mag = np.array(datadir[4],dtype=np.float64)
    lc_mag[extra_noise_index] += extra_noise
    lc_time = np.array(datadir[2],dtype=np.float64)
    lc_sig = np.array(datadir[5],dtype=np.float64)

    if residual:
        lc_single = np.array(datadir[7],dtype=np.float64)
        lc_mag = ((lc_single-lc_mag)/(lc_sig))**2

    lc_mag = (np.mean(lc_mag)-lc_mag)/np.std(lc_mag)
    lc_mag = lc_mag.reshape((500,1))
    
    lc_time = (lc_time-np.mean(lc_time))/np.std(lc_time)
    lc_time = lc_time.reshape((500,1))

    lc_sig = (lc_sig-np.mean(lc_sig))/np.std(lc_sig)
    lc_sig = lc_sig.reshape((500,1))
    # lc_sig = (lc_sig-lc_mean)/np.std(lc_sig)

    data_input = np.concatenate((lc_mag,lc_time,lc_sig),axis=1)

    lc_data = np.array([data_input])

    label = np.array(datadir[8]).astype(np.int64)

    return lc_data, label

def loader_fortest(data_root,posi_lc,num_skip=0,residual=False):
    ## argsdata: [u_0, rho, q, s, alpha, t_E, basis_m, t_0]
    ## args_singlefitting: [t_E,t_0,u_0,basis_m]
    ## [args_data, arg_singlefitting, time, d_time, lc_withnoi, err, lc_withoutnoi, lc_singlemodel, unet_label]

    datadir = list(np.load(data_root+str(posi_lc+num_skip)+".npy", allow_pickle=True))

    extra_noise_index = np.array(datadir[9],dtype=np.int)
    extra_noise = np.array(datadir[10],dtype=np.float64)

    lc_mag = np.array(datadir[4],dtype=np.float64)
    lc_time = np.array(datadir[2],dtype=np.float64)
    lc_sig = np.array(datadir[5],dtype=np.float64)
    lc_mag[extra_noise_index] += extra_noise
    if residual:
        lc_single = np.array(datadir[7],dtype=np.float64)
        lc_mag = ((lc_single-lc_mag)/(lc_sig))**2
    lc_mag = (np.mean(lc_mag)-lc_mag)/np.std(lc_mag)
    lc_mag = lc_mag.reshape((500,1))
    
    
    lc_time = (lc_time-np.mean(lc_time))/np.std(lc_time)
    lc_time = lc_time.reshape((500,1))


    
    lc_sig = (lc_sig-np.mean(lc_sig))/np.std(lc_sig)
    lc_sig = lc_sig.reshape((500,1))
    # lc_sig = (lc_sig-lc_mean)/np.std(lc_sig)

    data_input = np.concatenate((lc_mag,lc_time,lc_sig),axis=1)

    lc_data = np.array([data_input])

    label = np.array([datadir[8],datadir[2],datadir[4],datadir[5],datadir[6],datadir[7]])

    return lc_data, label, [datadir[9],datadir[10]]


def loader_grouptest(data_root,posi_lc,num_skip=0,residual=False):
    ## argsdata: [u_0, rho, q, s, alpha, t_E, basis_m, t_0]
    ## args_singlefitting: [t_E,t_0,u_0,basis_m]
    ## [args_data, arg_singlefitting, time, d_time, lc_withnoi, err, lc_withoutnoi, lc_singlemodel, unet_label]

    datadir = list(np.load(data_root+str(posi_lc+num_skip)+".npy", allow_pickle=True))

    extra_noise_index = np.array(datadir[9],dtype=np.int)
    extra_noise = np.array(datadir[10],dtype=np.float64)

    lc_mag = np.array(datadir[4],dtype=np.float64)
    lc_time = np.array(datadir[2],dtype=np.float64)
    lc_sig = np.array(datadir[5],dtype=np.float64)
    lc_mag[extra_noise_index] += extra_noise
    if residual:
        lc_single = np.array(datadir[7],dtype=np.float64)
        lc_mag = ((lc_single-lc_mag)/(lc_sig))**2
    lc_mag = (np.mean(lc_mag)-lc_mag)/np.std(lc_mag)
    lc_mag = lc_mag.reshape((500,1))
    
    
    lc_time = (lc_time-np.mean(lc_time))/np.std(lc_time)
    lc_time = lc_time.reshape((500,1))
    
    lc_sig = (lc_sig-np.mean(lc_sig))/np.std(lc_sig)
    lc_sig = lc_sig.reshape((500,1))
    # lc_sig = (lc_sig-lc_mean)/np.std(lc_sig)

    data_input = np.concatenate((lc_mag,lc_time,lc_sig),axis=1)

    lc_data = np.array([data_input])

    label = np.array([datadir[8],datadir[2],datadir[4],datadir[5],datadir[6],datadir[7]])

    return lc_data, label


def sample_curve(time,mag,err,length_resample=1000):
    if len(time) < length_resample:
        raise RuntimeError("The length of lightcurve must be %d"%size_check)
    new_order = random.sample(range(len(time)),length_resample)
    new_order = np.sort(new_order)
    return time[new_order],mag[new_order],err[new_order]


def loader_transform(time,mag,err,size_check=500):
    ## argsdata: [u_0, rho, q, s, alpha, t_E, basis_m, t_0]
    ## args_singlefitting: [t_E,t_0,u_0,basis_m]
    ## [args_data, arg_singlefitting, time, d_time, lc_withnoi, err, lc_withoutnoi, lc_singlemodel, unet_label]

    if len(time) != size_check:
        raise RuntimeError("The length of lightcurve must be %d"%size_check)

    lc_mag = np.array(mag,dtype=np.float64)
    lc_mag = (np.mean(lc_mag)-lc_mag)/np.std(lc_mag)
    lc_mag = lc_mag.reshape((500,1))
    
    lc_time = np.array(time,dtype=np.float64)
    lc_time = (lc_time-np.mean(lc_time))/np.std(lc_time)
    lc_time = lc_time.reshape((500,1))


    lc_sig = np.array(err,dtype=np.float64)
    lc_sig = (lc_sig-np.mean(lc_sig))/np.std(lc_sig)
    lc_sig = lc_sig.reshape((500,1))
    # lc_sig = (lc_sig-lc_mean)/np.std(lc_sig)

    data_input = np.concatenate((lc_mag,lc_time,lc_sig),axis=1)

    lc_data = np.array([data_input])

    return lc_data

class Mydataset(Dataset):
    def __init__(self,transform=None,target_transform=None,n_lc=None,data_root=None,num_skip=0,loader=default_loader,residual=False):
        self.n_lc = n_lc
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.data_root = data_root
        self.num_skip=num_skip
        self.residual=residual

    def __getitem__(self, index):
        lc, label = self.loader(self.data_root,index,self.num_skip,self.residual)
        if self.transform is not None:
            lc = self.transform(lc)
        return lc, label

    def __len__(self):
        return self.n_lc

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# components of unet

class double_conv1d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1):
        super(double_conv1d_bn,self).__init__()
        self.conv1 = nn.Conv1d(in_channels,out_channels,
                               kernel_size=kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.conv2 = nn.Conv1d(out_channels,out_channels,
                              kernel_size = kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out
    
class deconv1d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,strides=2,output_padding=0):
        super(deconv1d_bn,self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels,out_channels,
                                        kernel_size = kernel_size,
                                       stride = strides,output_padding=output_padding,bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()

        self.rnn = nn.GRU(
            input_size=3,
            hidden_size=3,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        # (batch,1000,3) -> (batch,1000,3) -> (batch,3000) -> 
        self.rnnout1 = nn.Linear(3000,2000)
        self.rnnout2 = nn.Linear(2000,500)

        self.layer1_conv = double_conv1d_bn(1,8)
        self.layer2_conv = double_conv1d_bn(8,16)
        self.layer3_conv = double_conv1d_bn(16,32)
        self.layer4_conv = double_conv1d_bn(32,64)
        self.layer5_conv = double_conv1d_bn(64,128)
        self.layer6_conv = double_conv1d_bn(128,64)
        self.layer7_conv = double_conv1d_bn(64,32)
        self.layer8_conv = double_conv1d_bn(32,16)
        self.layer9_conv = double_conv1d_bn(16,8)
        self.layer10_conv = nn.Conv1d(8,1,kernel_size=3,
                                     stride=1,padding=1,bias=True)
        
        self.deconv1 = deconv1d_bn(128,64)# ,output_padding=1
        self.deconv2 = deconv1d_bn(64,32,output_padding=1)
        self.deconv3 = deconv1d_bn(32,16)
        self.deconv4 = deconv1d_bn(16,8)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = x.view(-1,500,3)

        self.rnn.flatten_parameters()
        x,_ = self.rnn(x,None)

        x = x.contiguous().view(-1,1,3000)
        x = self.rnnout1(x)
        x = self.rnnout2(x)

        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool1d(conv1,2)
        
        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool1d(conv2,2)
        
        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool1d(conv3,2)
        
        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool1d(conv4,2)
        
        conv5 = self.layer5_conv(pool4)
        
        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1,conv4],dim=1)
        conv6 = self.layer6_conv(concat1)
        
        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2,conv3],dim=1)
        conv7 = self.layer7_conv(concat2)
        
        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3,conv2],dim=1)
        conv8 = self.layer8_conv(concat3)
        
        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4,conv1],dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = self.layer10_conv(conv9)
        outp = self.sigmoid(outp)
        return torch.cat([outp,1-outp],dim=1)
