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

import netmodule.netGRUiden as lcnet

name_group = "10to15"
use_gpu = torch.cuda.is_available()

# path & args
storedir_base = "/scratch/zerui603/KMT_simu_lowratio/qseries/"+name_group+"testbins/"
preload_Netmodel = "GRUresnet_iden_res_"+name_group+".pkl"
path_params = "/scratch/zerui603/netparams/"


# build histbins
## bins of mag: 8 ; bins of lgdchi: 8
magnitude_bins = np.linspace(18,22,9)
lgdchis_bins = np.linspace(1,5,9)

num_bins_mag = 8
num_bins_lgdchis = 8

num_eachbin = 1000

correct_matrix = np.zeros((num_bins_mag,num_bins_lgdchis,))

for i in range(num_bins_mag):
    for j in range(num_bins_lgdchis):
        network = lcnet.ResNet()
        criterion = nn.BCELoss()
        if use_gpu:
            # network = network.cuda()
            criterion = criterion.cuda()
            network = nn.DataParallel(network).cuda()

        network.load_state_dict(torch.load(path_params+preload_Netmodel))

        storedir = storedir_base + str(i)+"_"+str(j)+"/"

        input_batch = []
        label_batch = []
        chi_s_batch = []

        for index_npy in range(num_eachbin):
            lc_data,args = lcnet.default_loader_fortest(data_root = storedir,posi_lc = index_npy)
            
            input_batch.append(lc_data)

            dchi_s = args[-2]
            bslabel = args[-1]

            
            if (bslabel>0.5)&(dchi_s < 20):
                bslabel = 0
            
            label_batch.append(bslabel)
            chi_s_batch.append(dchi_s)

        input_batch = torch.from_numpy(np.array(input_batch)).float()
        if use_gpu:
            input_batch = input_batch.cuda()
        
        network.eval()
        output_batch = network(input_batch).detach().cpu().numpy()
        bspre_batch = np.around(output_batch.T[0])

        print(i,j,np.sum(bspre_batch))

        correct_matrix[i][j] = np.sum(bspre_batch)/num_eachbin

mag_forfig = np.linspace(18.25,21.75,8)
lgdchis_forfig = np.linspace(1.25,4.75,8)

plt.figure()
plt.yticks(range(len(mag_forfig)))
plt.gca().set_yticklabels(mag_forfig)
plt.xticks(range(len(lgdchis_forfig)))
plt.gca().set_xticklabels(lgdchis_forfig)
plt.imshow(correct_matrix, cmap=plt.cm.hot_r)
plt.colorbar()
plt.xlabel("$\log_{10} \Delta \chi^2$(single fitting)")
plt.ylabel("$m_0$")
plt.title(name_group)
plt.savefig("test_mag_lgdchis_"+name_group+".png")

