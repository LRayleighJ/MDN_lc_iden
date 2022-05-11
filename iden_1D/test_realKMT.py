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

import netmodule.netGRUiden as lcnet
import datamodule.datamethod as dm

##args

year_KMT = 2019
KMT_size = 3303

num_code = np.int(sys.argv[1])
forbidden_numlist = []# [1,2,3,4,5,6,7]

for forbidden_num in forbidden_numlist:
    if num_code == forbidden_num:
        exit()

name_group_list = ["00to05","05to10","10to15","15to20","20to25","25to30","30to35","35to40"]
name_group_test_list = ["00to05test","05to10test","10to15test","15to20test","20to25test","25to30test","30to35test","35to40test"]
name_group = name_group_list[num_code]

print("Name Group: ",name_group)

datapath_KMT = "/scratch/zerui603/KMT_dataprocess/"
preload_Netmodel = "GRUresnet_iden_res_"+name_group+".pkl"
path_params = "/scratch/zerui603/netparams/"

def renormal_data(x):
    return (x-np.mean(x))/np.std(x)

def KMTreal_dataloader(datapath, year, posi):
    datadir = list(np.load(datapath+"%04d%04d.npy"%(year,posi,), allow_pickle=True))
    # [[year,posi],time_new,chi_s_array,err_new]

    lc_mag = np.array(datadir[2],dtype=np.float64)**2
    # lc_mag = np.mean(np.sort(lc_mag)[-50:])-np.array(lc_mag)
    lc_mag = lc_mag.reshape((1000,1))
    lc_mag = renormal_data(lc_mag)
    
    lc_time = np.array(datadir[1],dtype=np.float64)
    lc_time = lc_time.reshape((1000,1))
    lc_time = renormal_data(lc_time)

    lc_sig = np.array(datadir[3],dtype=np.float64)
    lc_sig = lc_sig.reshape((1000,1))
    # lc_sig = (lc_sig-lc_mean)/np.std(lc_sig)
    lc_sig = renormal_data(lc_sig)

    data_input = np.concatenate((lc_mag,lc_time,lc_sig),axis=1)

    lc_data = np.array([data_input])

    return lc_data, label


def testnet(net_name,testsize):
    network = lcnet.ResNet()
    criterion = nn.BCELoss()
    if use_gpu:
        criterion = criterion.cuda()
        network = nn.DataParallel(network).cuda()

    network.load_state_dict(torch.load(path_params+preload_Netmodel))

    name_list = []
    input_batch = []

    for i in range(testsize):
        try:
            data_event = KMTreal_dataloader(datapath_KMT,year_KMT,i)
            input_batch.append(data_event)
            name_list.append(i)
        except:
            print(i, " error")
            continue

    input_batch = torch.from_numpy(np.array(input_batch)).float()
    if use_gpu:
        input_batch = input_batch.cuda()
    
    network.eval()
    output_batch = network(input_batch).detach().cpu().numpy()
    bspre_batch = np.around(output_batch.T[0])

    with open("KMTcatalog%04d"%(year_KMT,)+name_group+".dat","w") as file_info:
        for i in range(len(name_list)):
            if bspre_batch[i]>0.5
            print(name_list[i]," binary",file=file_info)
        
if __name__ == "__main__":
    testnet(name_group,KMT_size)


