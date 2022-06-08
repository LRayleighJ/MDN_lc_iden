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

import netmodule.unetforkmt as lcnet
import netmodule.netGRUiden as lcresnet

def chis(x1,x2,sig,weight=1):
    return np.sum((np.array(x1)-np.array(x2))**2/np.array(sig)**2*weight)

def chis_array(x1,x2,sig,weight=1):
    return (np.array(x1)-np.array(x2))**2/np.array(sig)**2*weight

        

# reload
reload = 0
preload_Netmodel = "GRU_unet_lowratio.pkl"
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
n_epochs = 200
learning_rate = 5e-3
stepsize = 10# 7
gamma_0 = 0.7
momentum = 0.5

## path of trainingset and validationset

rootdir = "/scratch/zerui603/KMT_unet/low_ratio/training/"
rootval = "/scratch/zerui603/KMT_unet/low_ratio/val/"


def test_distribution(paramsid,droppoint=0):
    thres = 1-0.9985
    testsize = 10000
    testsize_batch = 1000
    pre_0 = 0
    label_0 = 0
    preandlabel_0 = 0
    fig_path = "/home/zerui603/MDN_lc_iden/unet/testfig/"
    '''
    network = lcnet.ResNet()
    if use_gpu:
        network = nn.DataParallel(network).cuda()
    preload_Netmodel = "GRUresnet_iden_res_mix.pkl"
    path_params = "/scratch/zerui603/netparams/"
    network.load_state_dict(torch.load(path_params+preload_Netmodel))
    '''
    network = lcnet.Unet()
    resnet = lcresnet.ResNet()
    if use_gpu:
        network = nn.DataParallel(network).cuda()
        resnet = nn.DataParallel(resnet).cuda()
    network.load_state_dict(torch.load(path_params+preload_Netmodel[:-4]+"_"+str(paramsid)+".pkl"))
    
    preload_Netmodel_resnet = "GRUresnet_iden_res_mix.pkl"
    path_params_resnet = "/scratch/zerui603/netparams/"
    resnet.load_state_dict(torch.load(path_params_resnet+preload_Netmodel_resnet))

    network.eval()
    resnet.eval()

    valdata = lcnet.Mydataset(n_lc=testsize,data_root=rootval,num_skip=0,loader=lcnet.loader_fortest)
    valset = lcnet.DataLoaderX(valdata, batch_size=testsize_batch,shuffle=False,num_workers=num_process,pin_memory=True)

    valdata_resnet = lcresnet.Myresdataset(n_lc=testsize,data_root=rootval,loader=lcresnet.default_loader_fortest)
    valset_resnet = lcresnet.DataLoaderX(valdata_resnet, batch_size=testsize_batch,shuffle=False,num_workers=num_process,pin_memory=True)

    dchis_label_list = []
    dchis_label_origin_list = []
    dchis_predict_list = []
    dchis_origin_list = []

    rate_cover_list = []
    rate_wide_list = []

    length_label = []
    length_predict = []

    binaryclassi_list = []
    bspretest_list = []

    count_bias = 0
    count_plot = 0

    with torch.no_grad():
        # test binary classification
        for j,valdata in enumerate(valset_resnet):
            val_inputs, val_labelanddata = valdata
            val_inputs = val_inputs.float()
            
            if use_gpu:
                val_inputs = val_inputs.cuda()
            val_outputs = resnet(val_inputs).detach().cpu().numpy()
            bspre_batch = val_outputs.T[0]

            binaryclassi_list = np.append(binaryclassi_list,bspre_batch)

        for j,valdata in enumerate(valset):
            val_inputs, val_labelanddata = valdata
            val_labels = val_labelanddata[:,0,:].long()
            val_dataori = val_labelanddata[:,1:,:].numpy()
            val_inputs = val_inputs.float()
            if use_gpu:
                val_inputs = val_inputs.cuda()
                val_labels = val_labels.cuda()
            val_outputs = network(val_inputs).detach().cpu().numpy()
            val_inputs = val_inputs.detach().cpu().numpy()
            val_labels = val_labels.detach().cpu().numpy()

            for i in range(testsize_batch):
                lc_data = val_inputs[i][0].T
                predict = val_outputs[i][0]
                label  = val_labels[i]

                time,lc_withnoi,err,lc_withoutnoi,lc_singlemodel = val_dataori[i]

                predict_01_array = predict<=thres

                dchis_label_array = chis_array(lc_withnoi,lc_singlemodel,err,label)-chis_array(lc_withnoi,lc_withoutnoi,err,label)
                dchis_predict_array = chis_array(lc_withnoi,lc_singlemodel,err,predict_01_array)-chis_array(lc_withnoi,lc_withoutnoi,err,predict_01_array)
                
                dchis_label_origin = np.abs(np.sum(dchis_label_array))
                
                droprate = 0.01*droppoint
                # dchis_label_array = np.sort(dchis_label_array)[:-droppoint]# [:np.int((1-droprate)*len(dchis_label_array))]
                # dchis_predict_array = np.sort(dchis_predict_array)[:np.int((1-droprate)*len(dchis_predict_array))]

                dchis_label = np.abs(np.sum(dchis_label_array))
                dchis_predict = np.abs(np.sum(dchis_predict_array))

                if (dchis_predict <= 1)&(dchis_label<=1):
                    preandlabel_0 += 1
                    continue
                if (dchis_predict <= 1)|(dchis_label<=1):
                    if dchis_predict <= 1:
                        pre_0 += 1
                    if dchis_label<=1:
                        label_0 += 1
                    continue

                # testfig
                
                dchis_label_list.append(dchis_label)
                dchis_predict_list.append(dchis_predict)
                dchis_label_origin_list.append(dchis_label_origin)
                dchis_origin_list.append(np.abs(chis(lc_withnoi,lc_singlemodel,err)-chis(lc_withnoi,lc_withoutnoi,err)))
                rate_cover_list.append(np.sum(predict_01_array*label)/np.sum(label))
                rate_wide_list.append(np.sum(predict_01_array*label)/np.sum(predict_01_array))

                length_label.append(np.sum(label))
                length_predict.append(np.sum(predict_01_array))

                bspretest_list.append(binaryclassi_list[j*testsize_batch+i])

    bspretest_list = np.array(bspretest_list)
    print(preandlabel_0,pre_0,label_0,len(dchis_label_list))

    # draw origin line

    line50up,line50down,linexaxis1 = dm.get_rate_updown_line(0.5,np.log10(dchis_label_list),np.log10(dchis_predict_list),np.linspace(0,8,32))
    line75up,line75down,linexaxis2 = dm.get_rate_updown_line(0.75,np.log10(dchis_label_list),np.log10(dchis_predict_list),np.linspace(0,8,32))
    line90up,line90down,linexaxis3 = dm.get_rate_updown_line(0.9,np.log10(dchis_label_list),np.log10(dchis_predict_list),np.linspace(0,8,32))
    print(linexaxis1.shape,linexaxis2.shape,linexaxis3.shape)
    plt.figure(figsize=(8,8))
    # plt.subplot(211)
    # plt.title("$\log_{10}|\Delta \chi^2|$")
    plt.scatter(np.log10(dchis_label_list),np.log10(dchis_predict_list),s=8,zorder=3,alpha=0.2,c="blue")# [bspretest_list>0.85] ,label="binary"
    # plt.scatter(np.log10(dchis_label_list)[bspretest_list<0.85],np.log10(dchis_predict_list)[bspretest_list<0.85],s=8,zorder=3,alpha=0.5,c="red",label="1S1L")
    plt.plot(np.linspace(0,8,10),np.linspace(0,8,10),alpha=0.5,ls="--",c="black")
    plt.fill_between(linexaxis1,line50down,line50up, where=line50down<line50up, facecolor="orange",alpha=0.5,label="$<50\%$")
    plt.fill_between(linexaxis2,line75down,line50down, where=line75down<line50down,facecolor="greenyellow", alpha=0.5)
    plt.fill_between(linexaxis2,line50up,line75up, where=line50up<line75up, facecolor="greenyellow",alpha=0.5,label="$50\sim 75\%$")
    plt.fill_between(linexaxis3,line90down,line75down, where=line90down<line75down, facecolor="tomato", alpha=0.5)
    plt.fill_between(linexaxis3,line75up,line90up, where=line75up<line90up, facecolor="tomato", alpha=0.5,label="$75\sim 90\%$")
    plt.axis("scaled")
    plt.xlim((0,7.5))
    plt.ylim((0,7.5))
    plt.tick_params(labelsize=15)
    plt.xlabel("$\log_{10}|\Delta \chi^2|$(label)",fontsize=20)
    plt.ylabel("$\log_{10}|\Delta \chi^2|$(predicted)",fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/dchis2D_resnet_%d.pdf"%(droppoint,))
    plt.close()
    

    # plt.subplot(212)
    plt.figure(figsize=(8,8))
    # plt.title("$\log_{10}|\Delta \chi^2|/length$")
    length_label = np.array(length_label)
    length_predict = np.array(length_predict)

    length_label = (np.abs(length_label)-0.5)+0.5
    length_predict = (np.abs(length_predict)-0.5)+0.5

    line50up,line50down,linexaxis1 = dm.get_rate_updown_line(0.5,np.log10(dchis_label_list/length_label),np.log10(dchis_predict_list/length_predict),np.linspace(-1,5.5,32))
    line75up,line75down,linexaxis2 = dm.get_rate_updown_line(0.75,np.log10(dchis_label_list/length_label),np.log10(dchis_predict_list/length_predict),np.linspace(-1,5.5,32))
    line90up,line90down,linexaxis3 = dm.get_rate_updown_line(0.9,np.log10(dchis_label_list/length_label),np.log10(dchis_predict_list/length_predict),np.linspace(-1,5.5,32))
    print(linexaxis1.shape,linexaxis2.shape,linexaxis3.shape)
    
    plt.scatter(np.log10(dchis_label_list/length_label),np.log10(dchis_predict_list/length_predict),s=8,zorder=3,c="blue",alpha=0.2)
    # plt.scatter(np.log10(dchis_label_list/length_label)[bspretest_list<0.85],np.log10(dchis_predict_list/length_predict)[bspretest_list<0.85],s=8,zorder=3,c="red",alpha=0.5,label="1S1L")
    plt.plot(np.linspace(-1,6,10),np.linspace(-1,6,10),alpha=0.5,ls="--",c="black")
    plt.fill_between(linexaxis1,line50down,line50up, where=line50down<line50up, facecolor="orange",alpha=0.5,label="$<50\%$")
    plt.fill_between(linexaxis2,line75down,line50down, where=line75down<line50down,facecolor="greenyellow", alpha=0.5,label="$50\sim 75\%$")
    plt.fill_between(linexaxis2,line50up,line75up, where=line50up<line75up, facecolor="greenyellow",alpha=0.5)
    plt.fill_between(linexaxis3,line90down,line75down, where=line90down<line75down, facecolor="tomato", alpha=0.5,label="$75\sim 90\%$")
    plt.fill_between(linexaxis3,line75up,line90up, where=line75up<line90up, facecolor="tomato", alpha=0.5)
    plt.axis("scaled")
    plt.xlim(-1,6)
    plt.ylim(-1,6)
    plt.xlabel("$\log_{10}|\Delta \chi^2|/length$(label)",fontsize=20)
    plt.ylabel("$\log_{10}|\Delta \chi^2|/length$(predicted)",fontsize=20)
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=20)
    # plt.suptitle("Drop %d points"%(droppoint,))
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/dchis2Ds_resnet_%d.pdf"%(droppoint,))
    plt.close()
      

if __name__=="__main__":
    test_distribution(paramsid=40,droppoint=0)