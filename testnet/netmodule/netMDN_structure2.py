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
k_size1 = 5
k_size2 = 5
k_size3 = 5


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
    lc_data = np.array([lc_mag])

    return lc_data, label*5


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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_filters[0],
                out_channels=n_filters[1],
                kernel_size=k_size1
                ),
            nn.BatchNorm1d(n_filters[1]),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_filters[1],
                out_channels=n_filters[2],
                kernel_size=k_size1
                ),
            nn.BatchNorm1d(n_filters[2]),
            nn.ReLU()
        )
        self.bn1 = nn.BatchNorm1d(n_filters[2])
        self.mp1 = nn.MaxPool1d(2)
        self.conv2_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_filters[2],
                out_channels=n_filters[3],
                kernel_size=k_size1
            ),
            nn.BatchNorm1d(n_filters[3]),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_filters[3],
                out_channels=n_filters[4],
                kernel_size=k_size1
            ),
            nn.BatchNorm1d(n_filters[4]),
            nn.ReLU()
        )
        self.bn2 = nn.BatchNorm1d(n_filters[4])
        self.mp2 = nn.MaxPool1d(2)
        self.conv3_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_filters[4],
                out_channels=n_filters[5],
                kernel_size=k_size2
            ),
            nn.BatchNorm1d(n_filters[5]),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_filters[5],
                out_channels=n_filters[6],
                kernel_size=k_size2
            ),
            nn.BatchNorm1d(n_filters[6]),
            nn.ReLU()
        )
        self.bn3 = nn.BatchNorm1d(n_filters[6])
        self.mp3 = nn.MaxPool1d(2)
        self.conv4_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_filters[6],
                out_channels=n_filters[7],
                kernel_size=k_size2
            ),
            nn.BatchNorm1d(n_filters[7]),
            nn.ReLU()
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_filters[7],
                out_channels=n_filters[8],
                kernel_size=k_size2
            ),
            nn.BatchNorm1d(n_filters[8]),
            nn.ReLU()
        )
        self.bn4 = nn.BatchNorm1d(n_filters[8])
        self.mp4 = nn.MaxPool1d(2)
        self.conv5_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_filters[8],
                out_channels=n_filters[9],
                kernel_size=k_size3
            ),
            nn.BatchNorm1d(n_filters[9]),
            nn.ReLU()
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_filters[9],
                out_channels=n_filters[10],
                kernel_size=k_size3
            ),
            nn.BatchNorm1d(n_filters[10]),
            nn.ReLU()
        )
        self.mp5 = nn.MaxPool1d(2)
        self.conv6_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_filters[10],
                out_channels=n_filters[11],
                kernel_size=k_size3
            ),
            nn.BatchNorm1d(n_filters[11]),
            nn.ReLU()
        )
        self.conv6_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_filters[11],
                out_channels=n_filters[12],
                kernel_size=k_size3
            ),
            nn.BatchNorm1d(n_filters[12]),
            nn.ReLU()
        )

        self.bn5 = nn.BatchNorm1d(n_filters[12])
        self.mp6 = nn.MaxPool1d(2)

        # linear mapping

        self.fcarg1 = nn.Sequential(
            nn.Linear(11*512, 4*512),
            nn.Linear(4*512,2*512),
            nn.Linear(2*512,512),
            nn.Linear(512,256)
        )

        self.fcarg2 = nn.Sequential(
            nn.Linear(11*512, 4*512),
            nn.Linear(4*512,2*512),
            nn.Linear(2*512,512),
            nn.Linear(512,256)
        )

        self.fcarg3 = nn.Sequential(
            nn.Linear(11*512, 4*512),
            nn.Linear(4*512,2*512),
            nn.Linear(2*512,512),
            nn.Linear(512,256)
        )

        self.fcarg4 = nn.Sequential(
            nn.Linear(11*512, 4*512),
            nn.Linear(4*512,2*512),
            nn.Linear(2*512,512),
            nn.Linear(512,256)
        )

        self.opepi1 = nn.Linear(256,12)
        self.opepi2 = nn.Linear(256,12)
        self.opepi3 = nn.Linear(256,12)
        self.opepi4 = nn.Linear(256,12)

        self.opemu1 = nn.Linear(256,12)
        self.opemu2 = nn.Linear(256,12)
        self.opemu3 = nn.Linear(256,12)
        self.opemu4 = nn.Linear(256,12)

        self.opesigma1 = nn.Linear(256,12)
        self.opesigma2 = nn.Linear(256,12)
        self.opesigma3 = nn.Linear(256,12)
        self.opesigma4 = nn.Linear(256,12)

        self.lsfm = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        # x = self.mp1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.mp2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        #x = self.mp3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.mp4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        # x = self.mp5(x)
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        x = self.mp6(x)
        # print(x.shape)
        x = x.view(-1, 11*512)        

        pi1 = self.lsfm(self.opepi1(self.fcarg1(x)))
        pi2 = self.lsfm(self.opepi2(self.fcarg2(x)))
        pi3 = self.lsfm(self.opepi3(self.fcarg3(x)))
        pi4 = self.lsfm(self.opepi4(self.fcarg4(x)))

        mu1 = self.opemu1(self.fcarg1(x))
        mu2 = self.opemu2(self.fcarg2(x))
        mu3 = self.opemu3(self.fcarg3(x))
        mu4 = self.opemu4(self.fcarg4(x))

        sigma1 = torch.exp(self.opesigma1(self.fcarg1(x)))
        sigma2 = torch.exp(self.opesigma2(self.fcarg2(x)))
        sigma3 = torch.exp(self.opesigma3(self.fcarg3(x)))
        sigma4 = torch.exp(self.opesigma4(self.fcarg4(x)))
        
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