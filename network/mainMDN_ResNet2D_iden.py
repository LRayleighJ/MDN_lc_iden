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

import netmodule.netMDN_ResNet2D_iden as lcnet

# reload
reload = 0
preload_Netmodel = "resnet_params2D4args_iden.pkl"


# initialize GPU
use_gpu = torch.cuda.is_available()
N_gpu = 4

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

print(N_gpu)
print(os.cpu_count())
device_ids = list(range(N_gpu))


# torch.cuda.set_device(device_ids)
torch.backends.cudnn.benchmark = True

# define parameters
## number of points

## size of trainingset library
size_train = 600000
## size of validationset library
size_val = 20000

## batch size and epoch
batch_size_train = 3000
batch_size_val = 2000
n_epochs = 25
learning_rate = 5e-5
stepsize = 10
gamma_0 = 0.75
momentum = 0.3

## path of trainingset and validationset
rootdir = "/scratch/zerui603/KMTsimudata_iden/2Ddata/training/"
rootval = "/scratch/zerui603/KMTsimudata_iden/2Ddata/training/"


# Loading datas
trainingsdata = lcnet.Mydataset(n_lc=size_train,data_root=rootdir,prenum=0)
trainset = lcnet.DataLoaderX(trainingsdata, batch_size=batch_size_train,shuffle=True,num_workers=20,pin_memory=True)

valdata = lcnet.Mydataset(n_lc=size_val,data_root=rootval,prenum=size_train)
valset = lcnet.DataLoaderX(valdata, batch_size=batch_size_val,num_workers=20,pin_memory=True,shuffle=True)

# initialize model
network = lcnet.ResNet()
criterion = nn.BCELoss()
if use_gpu:
    network = network.cuda()
    criterion = criterion.cuda()
    network = nn.DataParallel(network,device_ids = device_ids)
if reload == 1:
    network.load_state_dict(torch.load(preload_Netmodel))

optimizer = optim.Adam(network.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=stepsize,gamma=gamma_0)
# optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

# Training

loss_figure = []
val_loss_figure = []
val_correct_list = []
cor_matrix = [[],[],[],[]]

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
        outputs = network(inputs)
        outputs = outputs.double()
        
        # print(np.max(outputs.detach().cpu().numpy()),np.min(outputs.detach().cpu().numpy()))
        # print(np.max(labels.detach().cpu().numpy()),np.min(labels.detach().cpu().numpy()))
        
        loss = criterion(outputs,labels)
    
        loss.backward()
        optimizer.step()
        # print("finish calculating",datetime.datetime.now())
        epoch_rs = epoch_rs + loss.detach().item()
        if sam%10 == 0:
            print("Epoch:[", epoch + 1, sam, "] loss:", loss.item(),str(datetime.datetime.now()))
        sam = sam+1
       
    scheduler.step()
    loss_figure.append(epoch_rs/sam)
    print("Training_Epoch:[", epoch + 1, "] Training_loss:", epoch_rs/sam,str(datetime.datetime.now()))

    if (epoch+1)%10 == 0:
        torch.save(network.state_dict(),preload_Netmodel)
        print("netparams have been saved once")

    gc.collect()
    

    val_epoch_rs = 0
    val_sam = 0
    val_correct = 0
    val_cor_00 = 0
    val_cor_01 = 0
    val_cor_10 = 0
    val_cor_11 = 0
    network.eval()
    with torch.no_grad():
        for j,valdata in enumerate(valset):
            val_inputs, val_labels = valdata
            val_inputs = val_inputs.float()
            if use_gpu:
                val_inputs = val_inputs.cuda()
                val_labels = val_labels.cuda()
            optimizer.zero_grad()
            val_outputs = network(val_inputs)
            val_outputs = val_outputs.double()
            loss = criterion(val_outputs,val_labels)
            val_sam = val_sam + 1
            val_epoch_rs = val_epoch_rs + loss.item()
            print("val:",val_sam,loss.item())

            val_outputs = np.around(val_outputs.cpu().detach().numpy())
            val_labels = val_labels.cpu().detach().numpy()
            # print(val_labels.shape)
            correct_num = np.sum(val_labels*val_outputs)
            ## 
            labels_0 = val_labels.T[0]
            labels_1 = val_labels.T[1]
            output_0 = val_outputs.T[0]
            output_1 = val_outputs.T[1]

            val_correct += correct_num
            val_cor_00 += np.sum(labels_0*output_0)
            val_cor_01 += np.sum(labels_0*output_1)
            val_cor_10 += np.sum(labels_1*output_0)
            val_cor_11 += np.sum(labels_1*output_1)



    val_loss_figure.append(val_epoch_rs/val_sam)
    val_correct_list.append(val_correct/size_val)
    print("val_Epoch:[", epoch + 1, "] val_loss:", val_epoch_rs/val_sam,str(datetime.datetime.now()))
    print("Correct valset: ",val_correct,"/",size_val)
    print("TT,TF,FT,FF",val_cor_00,val_cor_01,val_cor_10,val_cor_11)
    cor_matrix[0].append(val_cor_00)
    cor_matrix[1].append(val_cor_01)
    cor_matrix[2].append(val_cor_10)
    cor_matrix[3].append(val_cor_11)

    plt.figure(figsize=(18,18))
    plt.subplot(311)
    x = np.linspace(1,epoch+1,len(loss_figure))
    plt.plot(x, loss_figure,label = "training loss log-likehood")
    plt.plot(x, val_loss_figure,label = "val loss log-likehood")
    plt.title("loss-epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss BCELoss")
    plt.legend()

    plt.subplot(312)
    plt.plot(x, val_correct_list,label="accuracy")
    plt.title("Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")

    plt.subplot(313)
    plt.plot(x, cor_matrix[0],label="output:binary,label:binary")
    plt.plot(x, cor_matrix[1],label="output:single,label:binary")
    plt.plot(x, cor_matrix[2],label="output:binary,label:single")
    plt.plot(x, cor_matrix[3],label="output:single,label:single")
    plt.xlabel("epoch")
    plt.ylabel("Number")
    plt.legend()
    
    plt.savefig("loss_accuracy_Resnet2D.png")
    plt.close()

    


torch.save(network.state_dict(),preload_Netmodel)
