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
import pickle
import gc 

import netmodule.netGRU as lcnet

# reload
reload = 1
preload_Netmodel = "GRUresnet_params_4args.pkl"


# initialize GPU
use_gpu = torch.cuda.is_available()
N_gpu = 4

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

print(N_gpu)
print(os.cpu_count())
device_ids = list(range(N_gpu))
torch.backends.cudnn.benchmark = True

# define parameters
## number of points

## size of trainingset library
size_train = 1000000
## size of validationset library
size_val = 10000

## batch size and epoch
batch_size_train = 10000
batch_size_val =2000
n_epochs = 200
learning_rate = 3e-5
stepsize = 15
gamma_0 = 0.9
momentum = 0.5

## path of trainingset and validationset
rootdir = "/scratch/zerui603/KMTsimudata/training/"
rootval = "/scratch/zerui603/KMTsimudata/val2/"


# Loading datas
trainingsdata = lcnet.Mydataset(n_lc=size_train,data_root=rootdir,judge_train=0)
trainset = lcnet.DataLoaderX(trainingsdata, batch_size=batch_size_train,shuffle=True,num_workers=20,pin_memory=True)

valdata = lcnet.Mydataset(n_lc=size_val,data_root=rootval,judge_train=0)
valset = lcnet.DataLoaderX(valdata, batch_size=batch_size_val,num_workers=20,pin_memory=True,shuffle=True)

# initialize model
network = lcnet.ResNet()
criterion = lcnet.Loss_fn()
if use_gpu:
    network = network.cuda()
    criterion = criterion.cuda()
    network = nn.DataParallel(network,device_ids = device_ids)
if reload == 1:
    network.load_state_dict(torch.load(preload_Netmodel))

optimizer = optim.Adam(network.parameters(), lr=learning_rate)#, momentum = momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=stepsize,gamma=gamma_0)
# optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

# Training

loss_figure = []
val_loss_figure = []

for epoch in range(n_epochs):
    running_loss = 0.0
    epoch_rs = 0
    sam = 0
    network.train()
    print("start training",datetime.datetime.now())
    for (i,data) in enumerate(trainset):
        # print("start loading data",datetime.datetime.now())
        inputs, labels = data
        inputs = inputs.float()
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        # print("finish loading data",datetime.datetime.now())
        optimizer.zero_grad()
        outpi1,outpi2,outpi3,outpi4,outmu1,outmu2,outmu3,outmu4,outsigma1,outsigma2,outsigma3,outsigma4 = network(inputs)
        loss = criterion(labels,outpi1,outpi2,outpi3,outpi4,outmu1,outmu2,outmu3,outmu4,outsigma1,outsigma2,outsigma3,outsigma4)
        loss.backward()
        optimizer.step()
        # print("finish calculating",datetime.datetime.now())
        epoch_rs = epoch_rs + loss.detach().item()
        sam = sam+1
        print("Epoch:[", epoch + 1, sam, "] loss:", loss.item(),str(datetime.datetime.now()))
       
    scheduler.step()
    loss_figure.append(epoch_rs/sam)
    print("Training_Epoch:[", epoch + 1, "] Training_loss:", epoch_rs/sam,str(datetime.datetime.now()))
    

    val_epoch_rs = 0
    val_sam = 0
    network.eval()
    with torch.no_grad():
        for j,valdata in enumerate(valset):
            val_inputs, val_labels = valdata
            val_inputs = val_inputs.float()
            if use_gpu:
                val_inputs = val_inputs.cuda()
                val_labels = val_labels.cuda()
            optimizer.zero_grad()
            valoutpi1,valoutpi2,valoutpi3,valoutpi4,valoutmu1,valoutmu2,valoutmu3,valoutmu4,valoutsigma1,valoutsigma2,valoutsigma3,valoutsigma4 = network(val_inputs)
            loss = criterion(val_labels,valoutpi1,valoutpi2,valoutpi3,valoutpi4,valoutmu1,valoutmu2,valoutmu3,valoutmu4,valoutsigma1,valoutsigma2,valoutsigma3,valoutsigma4)
            val_sam = val_sam + 1
            val_epoch_rs = val_epoch_rs + loss.item()
            print("val:",val_sam,loss.item())
    val_loss_figure.append(val_epoch_rs/val_sam)
    print("val_Epoch:[", epoch + 1, "] val_loss:", val_epoch_rs/val_sam,str(datetime.datetime.now()))

    plt.figure()
    x = np.linspace(1,epoch+1,len(loss_figure))
    plt.plot(x, loss_figure,label = "training loss log-likelihood")
    plt.plot(x, val_loss_figure,label = "val loss log-likelihood")
    plt.title("loss-epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss log-likelihood")
    plt.legend()
    plt.savefig("loss.png")
    plt.close()

    if (epoch+1)%10 == 0:
        torch.save(network.state_dict(),preload_Netmodel)
        print("netparams have been saved once")

    gc.collect()


torch.save(network.state_dict(),preload_Netmodel)
