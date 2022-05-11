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

num_code = np.int(sys.argv[1])
forbidden_numlist = []

for forbidden_num in forbidden_numlist:
    if num_code == forbidden_num:
        exit()

name_group_list = ["00to05","05to10","10to15","15to20","20to25","25to30","30to35","35to40"]
name_group_test_list = ["00to05test","05to10test","10to15test","15to20test","20to25test","25to30test","30to35test","35to40test"]
name_group = name_group_list[num_code]

print("Name Group: ",name_group)

threshold_classi = [0.8787878787878789, 0.787878787878788, 0.888888888888889, 0.888888888888889, 0.888888888888889, 0.9090909090909092, 0.8080808080808082, 0.8686868686868687]
thres_net_test = threshold_classi[num_code]

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="4"

trainortest = 0 # 0:test, 1:train
fullorparttest = 2 # 0: part testfig 1: full testfig 2: no fig

# prepare

# reload
reload = 0
preload_Netmodel = "GRUresnet_iden_res_"+name_group+".pkl"
path_params = "/scratch/zerui603/netparams/"
num_process = 16

# initialize GPU
use_gpu = torch.cuda.is_available()

print(os.cpu_count())

# device_ids = [1,2,3,4]

# torch.cuda.set_device("cuda:4,5")

torch.backends.cudnn.benchmark = True

# define parameters
## number of points

## size of trainingset library
size_train = 1000000
## size of validationset library
size_val = 100000

## batch size and epoch
batch_size_train = 35000
batch_size_val =10000
n_epochs = 25
learning_rate_list = [8e-6,8e-7,1.5e-6,1e-6,1e-6,1e-6,2e-6,2e-6]#[0,1,2,3,4,5,6,7]
learning_rate = learning_rate_list[num_code] # 4e-6
stepsize = 5# 7
gamma_0 = 0.7
momentum = 0.5

## path of trainingset and validationset

rootdir = "/scratch/zerui603/KMT_simu_lowratio/qseries/"+name_group+"/"
rootval = "/scratch/zerui603/KMT_simu_lowratio/qseries/"+name_group+"test/"
rootdraw = "/home/zerui603/MDN_lc/iden_1D/testfig/"
fullrootdraw = "/scratch/zerui603/figtest_iden/"+name_group+"/"

## arguments for training
num_test = 100000
batch_test = 7500

def signlog(x):
    return np.log10(np.abs(x))

def testnet(datadir=rootval,fullorparttest = fullorparttest):
    network = lcnet.ResNet()
    criterion = nn.BCELoss()
    if use_gpu:
        criterion = criterion.cuda()
        network = nn.DataParallel(network).cuda()

    network.load_state_dict(torch.load(path_params+preload_Netmodel))

    num_batch = num_test//batch_test

    if num_test%batch_test != 0:
        num_batch += 1

    # making prediction
    structure_data = []
    m0_data = []

    for index_batch in range(num_batch):

        input_batch = []
        m0_batch = []
        structure_batch = []

        for index in range(index_batch*batch_test,np.min([(index_batch+1)*batch_test,num_test])):
            lc_data,args,extra_data = lcnet.default_loader_fortest(data_root = datadir,posi_lc = index,extra_index = [4,5,7])
            # [u_0, rho, q, s, alpha, t_E, basis_m, t_0, dchis, label]

            input_batch.append(lc_data)
            extra_data = np.array(extra_data)
            lc_sigma = extra_data[0]
            lc_binary_nonoi = extra_data[1]
            lc_single_nonoi = extra_data[2]
            dchi_s = args[-2]
            bslabel = args[-1]
            m0 = args[6]

            if (bslabel>0.5)&(dchi_s < 10):
                bslabel = 0

            structure_batch.append(np.sum((lc_binary_nonoi-lc_single_nonoi)**2))
            m0_batch.append(m0)
        
        input_batch = torch.from_numpy(np.array(input_batch)).float()
        if use_gpu:
            input_batch = input_batch.cuda()
        
        network.eval()
        output_batch = network(input_batch).detach().cpu().numpy()


        bspre_batch = output_batch.T[0]

        structure_data = np.append(structure_data,structure_batch)
        m0_data = np.append(m0_data,m0_batch)

    plt.figure(figsize=[20,15])
    plt.scatter(m0_data,np.log10(structure_data),s=1,alpha=0.2)
    plt.savefig("structure_m0_"+name_group+".png")
    plt.close()
        

        


if __name__=="__main__":
    testnet()
