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
import lmdb
import pickle
import gc 

## parameters of network
n_filters = [1,16,16,32,32,64,64,128,128,256,256,512,512]

def sigma_0(m):
    if m > 16.372:
        return 0.01*10**(0.34*(m-16.372))
    else:
        return 0.01*10**(0.17*(m-16.372))
    '''
    result = 0
    if m < 16:
        result = 0.01
    elif 16 <= m < 23:
        result = 10**(0.2*m-6.2)
    elif m >= 23:
        result = 10**(0.4*m-10.8)
    return result
    '''


def magnitude_tran(magni,m_0=18):
    return m_0 - 2.5*np.log10(magni)


sigma = np.frompyfunc(sigma_0, 1, 1)


def noise_model(magnitude):
    sigma_mag = sigma(magnitude)
    noi = np.random.randn(len(magnitude))*sigma_mag
    return noi.astype(np.float64),sigma_mag.astype(np.float64)


def chi_square(single,binary,sigma):
    return np.sum(np.power((single-binary)/sigma,2))


def default_loader(data_root,posi_lc,judge_train=0):
    # [u_0, rho, q, s, alpha, t_E]
    # datadir = pd.DataFrame(np.load(dataroot+str(posi_lc)+".npy", allow_pickle=True))
    datadir = list(np.load(data_root+str(posi_lc+1000000*judge_train)+".npy", allow_pickle=True))
    
    labels = np.array(datadir[0],dtype=np.float64)

    lc_mag = np.array(datadir[1],dtype=np.float64)

    lg_q = np.log10(labels[2])
    lg_s = np.log10(labels[3])
    alpha = labels[4]
    u0 = labels[0]

    q_label = lg_q/4
    s_label = (lg_s-np.log10(0.3))/(np.log10(3)-np.log10(0.3))
    # alpha_label = alpha/360
    # u0_label = u0
    ux_label = (u0*np.cos(np.pi/180*alpha)+1)/2
    uy_label = (u0*np.sin(np.pi/180*alpha)+1)/2
    

    label = np.array([-1.*float(q_label),float(s_label),float(ux_label),float(uy_label)]).astype(np.float64)

    return np.array([lc_mag]), label*5


class Mydataset(Dataset):
    def __init__(self,transform=None,target_transform=None,n_lc=None,data_root=None,judge_train=0,loader=default_loader):
        self.n_lc = n_lc
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.data_root = data_root
        self.judge_train = judge_train

    def __getitem__(self, index):
        lc, label = self.loader(self.data_root,index,self.judge_train)
        if self.transform is not None:
            lc = self.transform(lc)
        return lc, label

    def __len__(self):
        return self.n_lc


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# ResNet34
class ResBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride,shortcut=None):
        super(ResBlock,self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = shortcut

    def forward(self,x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return nn.ReLU(inplace=True)(out)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.pre = nn.Sequential(
            nn.Conv2d(1,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )

        self.body = self.makelayers([3,4,6,3])

        self.fc1 = nn.Linear(25*512, 2*864)
        self.fc2 = nn.Linear(2*864,2*512)
        self.fc3 = nn.Linear(2*512,2*256)
        self.fc4 = nn.Linear(2*256,144)

        self.opepi1 = nn.Linear(144,12)
        self.opepi2 = nn.Linear(144,12)
        self.opepi3 = nn.Linear(144,12)
        self.opepi4 = nn.Linear(144,12)

        self.opemu1 = nn.Linear(144,12)
        self.opemu2 = nn.Linear(144,12)
        self.opemu3 = nn.Linear(144,12)
        self.opemu4 = nn.Linear(144,12)

        self.opesigma1 = nn.Linear(144,12)
        self.opesigma2 = nn.Linear(144,12)
        self.opesigma3 = nn.Linear(144,12)
        self.opesigma4 = nn.Linear(144,12)

        self.lsfm = nn.LogSoftmax(dim=-1)
    
    def makelayers(self,blocklist):
        self.layers = []
        for index,blocknum in enumerate(blocklist):
            if index != 0:
                shortcut = nn.Sequential(
                    nn.Conv2d(64*2**(index-1),64*2**index,1,2,bias=False),
                    nn.BatchNorm2d(64*2**index)
                )
                self.layers.append(ResBlock(64*2**(index-1),64*2**index,2,shortcut))
            for _ in range(0 if index==0 else 1,blocknum):
                self.layers.append(ResBlock(64*2**index,64*2**index,1))
        return nn.Sequential(*self.layers)

    def forward(self, x):
        # print(x.shape)
        x = self.pre(x)
        x = self.body(x)
        # print(x.shape)
        x = x.view(-1,25*512)
        # print(x.shape)
        x = self.fc1(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        # x = F.dropout(x, p=0.25, training=self.training)
        x = self.fc4(x)

        pi1 = self.lsfm(self.opepi1(x))
        pi2 = self.lsfm(self.opepi2(x))
        pi3 = self.lsfm(self.opepi3(x))
        pi4 = self.lsfm(self.opepi4(x))

        mu1 = self.opemu1(x)
        mu2 = self.opemu2(x)
        mu3 = self.opemu3(x)
        mu4 = self.opemu4(x)

        sigma1 = torch.exp(self.opesigma1(x))
        sigma2 = torch.exp(self.opesigma2(x))
        sigma3 = torch.exp(self.opesigma3(x))
        sigma4 = torch.exp(self.opesigma4(x))
        
        return pi1,pi2,pi3,pi4,mu1,mu2,mu3,mu4,sigma1,sigma2,sigma3,sigma4
'''
    def loss_fn(self, y, pi1,pi2,pi3,pi4,mu1,mu2,mu3,mu4,sigma1,sigma2,sigma3,sigma4):
        mixture1 = torch.distributions.normal.Normal(mu1, sigma1)
        log_prob1 = mixture1.log_prob(y.t()[0].t())
        weighted_logprob1 = log_prob1 + pi1
        log_sum1 = torch.logsumexp(weighted_logprob1, dim=-1)

        mixture2 = torch.distributions.normal.Normal(mu2, sigma2)
        log_prob2 = mixture2.log_prob(y.t()[1].t())
        weighted_logprob2 = log_prob2 + pi2
        log_sum2 = torch.logsumexp(weighted_logprob2, dim=-1)

        mixture3 = torch.distributions.normal.Normal(mu3, sigma3)
        log_prob3 = mixture3.log_prob(y.t()[2].t())
        weighted_logprob3 = log_prob3 + pi3
        log_sum3 = torch.logsumexp(weighted_logprob3, dim=-1)

        mixture4 = torch.distributions.normal.Normal(mu4, sigma4)
        log_prob4 = mixture4.log_prob(y.t()[3].t())
        weighted_logprob4 = log_prob4 + pi4
        log_sum4 = torch.logsumexp(weighted_logprob4, dim=-1)
        
        return -log_sum1.mean()-log_sum2.mean()-log_sum3.mean()-log_sum4.mean()
'''

class Loss_fn(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,y,pi1,pi2,pi3,pi4,mu1,mu2,mu3,mu4,sigma1,sigma2,sigma3,sigma4):
        mixture1 = torch.distributions.normal.Normal(mu1, sigma1)
        # print(y.shape)
        # print(y.t()[0].t().shape)
        log_prob1 = mixture1.log_prob(y.t()[0].t().unsqueeze(-1))
        weighted_logprob1 = log_prob1 + pi1
        log_sum1 = torch.logsumexp(weighted_logprob1, dim=-1)

        mixture2 = torch.distributions.normal.Normal(mu2, sigma2)
        log_prob2 = mixture2.log_prob(y.t()[1].t().unsqueeze(-1))
        weighted_logprob2 = log_prob2 + pi2
        log_sum2 = torch.logsumexp(weighted_logprob2, dim=-1)

        mixture3 = torch.distributions.normal.Normal(mu3, sigma3)
        log_prob3 = mixture3.log_prob(y.t()[2].t().unsqueeze(-1))
        weighted_logprob3 = log_prob3 + pi3
        log_sum3 = torch.logsumexp(weighted_logprob3, dim=-1)

        mixture4 = torch.distributions.normal.Normal(mu4, sigma4)
        log_prob4 = mixture4.log_prob(y.t()[3].t().unsqueeze(-1))
        weighted_logprob4 = log_prob4 + pi4
        log_sum4 = torch.logsumexp(weighted_logprob4, dim=-1)
        
        return -log_sum1.mean()-log_sum2.mean()-log_sum3.mean()-log_sum4.mean()