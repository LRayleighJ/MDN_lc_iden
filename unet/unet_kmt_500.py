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
import multiprocessing as mp
import sys
import imageio

import datamodule.dm as dm
import netmodule.unetforkmt_500 as lcnet

def chis(x1,x2,sig,weight=1):
    return np.sum((np.array(x1)-np.array(x2))**2/np.array(sig)**2*weight)

def chis_array(x1,x2,sig,weight=1):
    return (np.array(x1)-np.array(x2))**2/np.array(sig)**2*weight

        

testsize=10000

# reload
reload = 0
preload_Netmodel = "GRU_unet_500.pkl"
path_params = "/scratch/zerui603/netparams/"
num_process = 16

# initialize GPU
use_gpu = torch.cuda.is_available()
print("GPU:", use_gpu)

if use_gpu:
    pass
else:
    print("GPU is unavailable")
    exit()

print(os.cpu_count())

# device_ids = [1,2,3,4]

# torch.cuda.set_device("cuda:4,5")

torch.backends.cudnn.benchmark = True

# define parameters
## number of points

## size of trainingset library
size_train = 200000
## size of validationset library
size_val = 20000

## batch size and epoch
batch_size_train = 10000
batch_size_val =1000
n_epochs = 250
learning_rate = 6e-4
stepsize = 5# 7
gamma_0 = 0.9
momentum = 0.5

## path of trainingset and validationset

rootdir = "/scratch/zerui603/KMT_unet/extra_noise/training/"
rootval = "/scratch/zerui603/KMT_unet/extra_noise/val/"

# training

def training(paramsid):
    # Loading datas
    trainingdata = lcnet.Mydataset(n_lc=size_train,data_root=rootdir)
    trainset = lcnet.DataLoaderX(trainingdata, batch_size=batch_size_train,shuffle=True,num_workers=num_process,pin_memory=True)

    valdata = lcnet.Mydataset(n_lc=size_val,data_root=rootval)
    valset = lcnet.DataLoaderX(valdata, batch_size=batch_size_val,shuffle=True,num_workers=num_process,pin_memory=True)

    # initialize model
    network = lcnet.Unet()

    weights = [1.0, 20.0]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # criterion = nn.CrossEntropyLoss()
    if use_gpu:
        # network = network.cuda()
        criterion = criterion.cuda()
        network = nn.DataParallel(network).cuda()
    if reload == 1:
        network.load_state_dict(torch.load(path_params+preload_Netmodel[:-4]+"_"+str(paramsid)+".pkl"))

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=stepsize,gamma=gamma_0)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True, threshold=0.05, threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-7)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1)
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
            
            inputs, labels = data
            inputs = inputs.float()
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()
            
            outputs = network(inputs)
            # outputs = outputs.double()
            
            loss = criterion(outputs,labels)
            # print(labels)
        
            loss.backward()
            optimizer.step()
            # print("finish calculating",datetime.datetime.now())
            epoch_rs = epoch_rs + loss.detach().item()
            
            if sam%1 == 0:
                print("Epoch:[", epoch + 1, sam, "] loss:", loss.item(),str(datetime.datetime.now()))
                
            sam = sam+1
        
        
        loss_figure.append(epoch_rs/sam)
        print("Training_Epoch:[", epoch + 1, "] Training_loss:", epoch_rs/sam,str(datetime.datetime.now()))
        print("learning rate: ",optimizer.state_dict()['param_groups'][0]['lr'])

        if (epoch+1)%2 == 0:
            torch.save(network.state_dict(),path_params+preload_Netmodel[:-4]+"_"+str(epoch+1)+".pkl")
            print("netparams have been saved once",epoch+1)

        gc.collect()

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
                val_outputs = network(val_inputs)
                # val_outputs = val_outputs.double()
                loss = criterion(val_outputs,val_labels)
                val_sam = val_sam + 1
                val_epoch_rs = val_epoch_rs + loss.item()
                print("val:",val_sam,loss.item())

        val_loss_figure.append(val_epoch_rs/val_sam)

        scheduler.step()


        print("val_Epoch:[", epoch + 1, "] val_loss:", val_epoch_rs/val_sam,str(datetime.datetime.now()))

        plt.figure()
        x = np.linspace(1,epoch+1,len(loss_figure))
        plt.plot(x, loss_figure,label = "training loss log-likelihood")
        plt.plot(x, val_loss_figure,label = "val loss log-likelihood")
        plt.title("loss-epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss BCELoss")
        plt.legend()
        
        plt.savefig("/home/zerui603/MDN_lc_iden/loss_accuracy_Unet_lowratio_500.png")
        plt.close()

    torch.save(network.state_dict(),path_params+preload_Netmodel)
    np.save("Unet_loss_lowratio_500.npy",np.array([loss_figure,val_loss_figure]))

def testfig(num_test,num_skip):
    for i in range(num_test):
        lc_data, lc_label, extra_noise_list = lcnet.loader_fortest(rootval,i,num_skip)
        extra_noise_index, extra_noise = extra_noise_list
        label,time,lc_withnoi,err,lc_withoutnoi,lc_singlemodel = lc_label

        lc_withnoi = np.array(lc_withnoi)
        extra_noise_index = np.array(extra_noise_index,dtype=np.int)
        extra_noise = np.array(extra_noise,dtype=np.float)

        lc_withnoi[extra_noise_index] += extra_noise

        s_point = lc_withnoi[label<0.5]
        s_time = time[label<0.5]
        b_point = lc_withnoi[label>0.5]
        b_time = time[label>0.5]


        plt.figure(figsize=(10,6))
        plt.scatter(s_time,s_point,s=4,alpha=0.5,label = "label no structure",c="blue")
        plt.scatter(b_time,b_point,s=4,alpha=0.5,label = "label with structure",c="tomato")
        plt.scatter(time[extra_noise_index],lc_withnoi[extra_noise_index],s=4,alpha=0.5,label = "extra noise",c="black")
        plt.plot(time,lc_withoutnoi,ls="--",label="binary model",c="green",alpha=0.3)
        plt.plot(time,lc_singlemodel,ls="--",label="single model",c="red",alpha=0.3)
        plt.xlabel("t",fontsize=16)
        plt.ylabel("Mag",fontsize=16)
        plt.legend()
        plt.gca().invert_yaxis()
        
        plt.savefig("/home/zerui603/MDN_lc_iden/unet/testfig/"+str(np.int(i))+".pdf")
        plt.close()

def test_threshold(paramsid=0):
    testsize = 10000
    testsize_batch = 500
    network = lcnet.Unet()
    if use_gpu:
        network = nn.DataParallel(network).cuda()
    
    network.load_state_dict(torch.load(path_params+preload_Netmodel[:-4]+"_"+str(paramsid)+".pkl"))

    network.eval()

    valdata = lcnet.Mydataset(n_lc=testsize,data_root=rootval,num_skip=0)
    valset = lcnet.DataLoaderX(valdata, batch_size=testsize_batch,shuffle=True,num_workers=num_process,pin_memory=True)

    network.eval()

    test_list_label = np.array([])
    test_list_output = np.array([])
    with torch.no_grad():
        for j,valdata in enumerate(valset):
            val_inputs, val_labels = valdata
            val_inputs = val_inputs.float()
            if use_gpu:
                val_inputs = val_inputs.cuda()
                val_labels = val_labels.cuda()
            val_outputs = network(val_inputs).detach().cpu().numpy()
            val_labels = val_labels.detach().cpu().numpy()

            for i in range(val_outputs.shape[0]):
                test_list_label = np.append(test_list_label,val_labels[i])
                test_list_output = np.append(test_list_output,val_outputs[i][1])

    print(test_list_label.shape)
    print(test_list_output.shape)

    thres_list = np.linspace(0,1,100)
    accuracy_list = []
    for thres_test in thres_list:
        accuracy_list.append(np.mean((test_list_output > thres_test)))

    plt.figure()
    plt.plot(thres_list,accuracy_list)
    plt.plot(thres_list,np.mean(test_list_label)*np.ones(thres_list.shape))
    plt.savefig("accu_thres_500.png")
    plt.close()

                

if __name__=="__main__":
    training(paramsid=0)
    # testfig(num_test=100,num_skip=0)
    # test_threshold(paramsid=70)