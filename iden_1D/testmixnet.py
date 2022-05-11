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

name_group_test_list = ["00to05test","05to10test","10to15test","15to20test","20to25test","25to30test","30to35test","35to40test"]

use_gpu = torch.cuda.is_available()

def testnet(name_group=""):
    num_test = 100000
    batch_test = 7500
    rootval = "/scratch/zerui603/KMT_simu_lowratio/qseries/"+name_group+"/"
    rootdraw = "/home/zerui603/MDN_lc/iden_1D/testfig/"
    fullrootdraw = "/scratch/zerui603/figtest_iden/"+name_group+"/"
    # mag_bins = np.linspace(16,20,17)
    # lgdchis_bins = np.linspace(1.5,3.5,17)
    # initialize model
    network = lcnet.ResNet()
    criterion = nn.BCELoss()
    if use_gpu:
        # network = network.cuda()
        criterion = criterion.cuda()
        network = nn.DataParallel(network).cuda()
    preload_Netmodel = "GRUresnet_iden_res_mix.pkl"
    path_params = "/scratch/zerui603/netparams/"
    network.load_state_dict(torch.load(path_params+preload_Netmodel))
    
    bspre_total = []
    label_total = []
    dchis_total = []

    num_batch = num_test//batch_test

    if num_test%batch_test != 0:
        num_batch += 1

    # making prediction
    for index_batch in range(num_batch):

        input_batch = []
        label_batch = []
        dchis_batch = []

        # pre_total_batch = []
        # chis_total_batch = []


        for index in range(index_batch*batch_test,np.min([(index_batch+1)*batch_test,num_test])):
            lc_data,args,_ = lcnet.default_loader_fortest(data_root = rootval,posi_lc = index)
            # [u_0, rho, q, s, alpha, t_E, basis_m, t_0, dchis, label]
            input_batch.append(lc_data)

            dchis = args[-2]
            bslabel = args[-1]
            m0 = args[6]

            
            if (bslabel>0.5)&(dchis < 10):
                bslabel = 0
            
            label_batch.append(bslabel)
            dchis_batch.append(dchis)
        
        input_batch = torch.from_numpy(np.array(input_batch)).float()
        if use_gpu:
            input_batch = input_batch.cuda()
        
        network.eval()
        output_batch = network(input_batch).detach().cpu().numpy()

        bspre_batch = output_batch.T[0]
        label_batch = np.array(label_batch).astype(np.int)

        bspre_total = np.append(bspre_total,bspre_batch)
        label_total = np.append(label_total,label_batch)
        dchis_total = np.append(dchis_total,dchis_batch)
    
    np.save("mixdatanet_"+name_group+".npy",np.array([bspre_total,label_total,dchis_total]))

    thres_list = np.linspace(0,0.99,100)
    threshold=0

    for thres in thres_list:
        FPrate = np.sum((bspre_total>thres)*(np.abs(dchis_total)<10))/np.sum(bspre_total>thres)
        if FPrate<0.01:
            print(name_group, thres)
            threshold = thres
            break

    # hist
    lgdchis_total = np.log10(np.abs(dchis_total))

    lgdchis_act0 = lgdchis_total[label_total<0.5]
    lgdchis_act1 = lgdchis_total[label_total>0.5]
    lgdchis_pre0 = lgdchis_total[bspre_total<threshold]
    lgdchis_pre1 = lgdchis_total[bspre_total>=threshold]

    plt.figure(figsize=[7,5])

    plt.hist(lgdchis_act1,bins=80,range=(-1,7),label="actual binary",histtype="step")
    plt.hist(lgdchis_act0,bins=80,range=(-1,7),label="actual single",histtype="step")
    plt.hist(lgdchis_pre1,bins=80,range=(-1,7),label="predicted binary",histtype="step")
    plt.hist(lgdchis_pre0,bins=80,range=(-1,7),label="predicted single",histtype="step")

    plt.xlabel("$\log_{10} |\Delta \chi^2|$")
    # plt.legend()
    # plt.title("range of $\log_{10}q$: (-%s.%s, -%s.%s )"%(name_group[-2],name_group[-1],name_group[0],name_group[1]),fontsize=30)
    plt.savefig("histbs_mixdatanet_"+name_group+".pdf")
    plt.close()

    return np.array([np.histogram(lgdchis_act0,bins=80,range=(-1,7)),np.histogram( lgdchis_act1,bins=80,range=(-1,7)),np.histogram( lgdchis_pre0,bins=80,range=(-1,7)),np.histogram( lgdchis_pre1,bins=80,range=(-1,7))])

if __name__ == "__main__":
    ratedata_list = []
    labeldata_list = []
    for i,name_group in enumerate(name_group_test_list):
        hist_data = testnet(name_group)
        ratedata_list.append(hist_data)
        labeldata_list.append("range of $\log_{10}q$: (-%s.%s, -%s.%s )"%(name_group[-2],name_group[-1],name_group[0],name_group[1]))
    np.save("ratedata_mixnet.npy",ratedata_list)

    hist_axis = np.linspace(-1,7,80)
    plt.figure(figsize=(10,6))
    for i,ratedata in enumerate(ratedata_list):
        hist_lgdchis_act0,hist_lgdchis_act1,hist_lgdchis_pre0,hist_lgdchis_pre1 = ratedata
        hist_lgdchis_act0 = np.array(hist_lgdchis_act0)
        hist_lgdchis_act0 = np.array(hist_lgdchis_act1)
        hist_lgdchis_pre0 = np.array(hist_lgdchis_pre0)
        hist_lgdchis_pre1 = np.array(hist_lgdchis_pre1)
        rate_line_0 = hist_lgdchis_pre0[hist_lgdchis_act0>0]/hist_lgdchis_act0[hist_lgdchis_act0>0]
        rate_line_1 = hist_lgdchis_pre1[hist_lgdchis_act1>0]/hist_lgdchis_act1[hist_lgdchis_act1>0]
        hist_axis_temp0 = hist_axis[hist_lgdchis_act0>0]
        hist_axis_temp1 = hist_axis[hist_lgdchis_act1>0]
        plt.plot(hist_axis_0, rate_line_0)
        plt.plot(hist_axis_1, rate_line_1,label=labeldata_list[i])
    plt.xlabel("$\log_{10} |\Delta \chi^2|$")
    plt.xlim((-1,4))
    plt.legend()
    plt.savefig("ratehistbs_mixdatanet.pdf")
    plt.close()

        

        

        
    