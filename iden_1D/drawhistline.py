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
forbidden_numlist = []# [0,1,2,3,4,5,6,7]

for forbidden_num in forbidden_numlist:
    if num_code == forbidden_num:
        exit()

name_group_list = ["00to05","05to10","10to15","15to20","20to25","25to30","30to35","35to40"]
name_group_test_list = ["00to05test","05to10test","10to15test","15to20test","20to25test","25to30test","30to35test","35to40test"]
name_group = name_group_list[num_code]

print("Name Group: ",name_group)

threshold_classi = [0.9292929292929294, 0.7171717171717172, 0.7272727272727273, 0.7474747474747475, 0.7171717171717172, 0.7373737373737375, 0.6161616161616162, 0.7171717171717172]
thres_net_test = threshold_classi[num_code]

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="4"

trainortest = 0 # 0:test, 1:train
fullorparttest = 2 # 0: part testfig 1: full testfig 2: no fig

# prepare

# reload
reload = 1
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
n_epochs = 80
learning_rate = 8e-6 # 4e-6
stepsize = 15# 7
gamma_0 = 0.75
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
    mag_bins = np.linspace(16,20,17)
    lgdchis_bins = np.linspace(1,5,17)
    # initialize model
    network = lcnet.ResNet()
    criterion = nn.BCELoss()
    if use_gpu:
        # network = network.cuda()
        criterion = criterion.cuda()
        network = nn.DataParallel(network).cuda()

    network.load_state_dict(torch.load(path_params+preload_Netmodel))

    b_pre = []
    s_pre = []

    bchi_s_pre = []
    bchi_s_act = []
    bm0_pre = []
    bm0_act = []
    bchi_s_act_true = []
    bm0_act_true = []

    schi_s_pre = []
    schi_s_act = []
    sm0_pre = []
    sm0_act = []
    schi_s_act_true = []
    sm0_act_true = []

    # file_actual/predicted
    file_bb = []
    file_bs = []
    file_sb = []
    file_ss = []

    num_batch = num_test//batch_test

    if num_test%batch_test != 0:
        num_batch += 1

    # making prediction
    for index_batch in range(num_batch):

        input_batch = []
        label_batch = []
        chi_s_batch = []
        file_batch = []
        m0_batch = []


        for index in range(index_batch*batch_test,np.min([(index_batch+1)*batch_test,num_test])):
            lc_data,args = lcnet.default_loader_fortest(data_root = datadir,posi_lc = index)
            # [u_0, rho, q, s, alpha, t_E, basis_m, t_0, dchis, label]
            input_batch.append(lc_data)

            file_batch.append(index)

            dchi_s = args[-2]
            bslabel = args[-1]
            m0 = args[6]

            
            if (bslabel>0.5)&(dchi_s < 10):
                bslabel = 0
            
            label_batch.append(bslabel)
            chi_s_batch.append(dchi_s)
            m0_batch.append(m0)
        
        input_batch = torch.from_numpy(np.array(input_batch)).float()
        if use_gpu:
            input_batch = input_batch.cuda()
        
        network.eval()
        output_batch = network(input_batch).detach().cpu().numpy()


        bspre_batch = output_batch.T[0]
        b_pre_batch = bspre_batch[label_batch>0.5]
        s_pre_batch = bspre_batch[label_batch<0.5]

        label_batch = np.array(label_batch).astype(np.int)

        file_batch = np.array(file_batch)

        file_bb_batch = file_batch[(label_batch>0.5)&(bspre_batch>=thres_net_test)]
        file_bs_batch = file_batch[(label_batch>0.5)&(bspre_batch<thres_net_test)]
        file_sb_batch = file_batch[(label_batch<0.5)&(bspre_batch>thres_net_test)]
        file_ss_batch = file_batch[(label_batch<0.5)&(bspre_batch<=thres_net_test)] 

        if index_batch == 0:
            b_pre = b_pre_batch
            s_pre = s_pre_batch
            bchi_s_pre = np.array(chi_s_batch)[np.argwhere(bspre_batch==1).T[0]].copy()
            bm0_pre = np.array(m0_batch)[np.argwhere(bspre_batch==1).T[0]].copy()
            bchi_s_act = np.array(chi_s_batch)[np.argwhere(label_batch==1).T[0]].copy()
            bchi_s_act_true = np.array(chi_s_batch)[np.argwhere((label_batch==1)&(bspre_batch==1)).T[0]].copy()
            bm0_act = np.array(m0_batch)[np.argwhere(label_batch==1).T[0]].copy()
            bm0_act_true = np.array(m0_batch)[np.argwhere((label_batch==1)&(bspre_batch==1)).T[0]].copy()
            schi_s_pre = np.array(chi_s_batch)[np.argwhere(bspre_batch==0).T[0]].copy()
            sm0_pre = np.array(m0_batch)[np.argwhere(bspre_batch==0).T[0]].copy()
            schi_s_act = np.array(chi_s_batch)[np.argwhere(label_batch==0).T[0]].copy()
            sm0_act = np.array(m0_batch)[np.argwhere(label_batch==0).T[0]].copy()

            file_bb = file_bb_batch.copy()
            file_bs = file_bs_batch.copy()
            file_sb = file_sb_batch.copy()
            file_ss = file_ss_batch.copy()

        else:
            b_pre = np.append(b_pre,b_pre_batch)
            s_pre = np.append(s_pre,s_pre_batch)
            bchi_s_pre = np.append(bchi_s_pre, np.array(chi_s_batch)[np.argwhere(bspre_batch==1).T[0]] )
            bchi_s_act = np.append(bchi_s_act, np.array(chi_s_batch)[np.argwhere(label_batch==1).T[0]] )
            bchi_s_act_true = np.append(bchi_s_act_true, np.array(chi_s_batch)[np.argwhere((label_batch==1)&(bspre_batch==1)).T[0]] )
            schi_s_pre = np.append(schi_s_pre, np.array(chi_s_batch)[np.argwhere(bspre_batch==0).T[0]] )
            schi_s_act = np.append(schi_s_act, np.array(chi_s_batch)[np.argwhere(label_batch==0).T[0]] )

            bm0_pre = np.append(bm0_pre, np.array(m0_batch)[np.argwhere(bspre_batch==1).T[0]] )
            bm0_act = np.append(bm0_act, np.array(m0_batch)[np.argwhere(label_batch==1).T[0]] )
            bm0_act_true = np.append(bm0_act_true, np.array(m0_batch)[np.argwhere((label_batch==1)&(bspre_batch==1)).T[0]] )
            sm0_pre = np.append(sm0_pre, np.array(m0_batch)[np.argwhere(bspre_batch==0).T[0]] )
            sm0_act = np.append(sm0_act, np.array(m0_batch)[np.argwhere(label_batch==0).T[0]] )

            file_bb = np.append(file_bb, file_bb_batch)
            file_bs = np.append(file_bs, file_bs_batch)
            file_sb = np.append(file_sb, file_sb_batch)
            file_ss = np.append(file_ss, file_ss_batch)
    print("label: 1, pre: 1", len(file_bb))
    print("label: 1, pre: 0", len(file_bs))
    print("label: 0, pre: 1", len(file_sb))
    print("label: 0, pre: 0", len(file_ss))
    
    # test bs bin

    mag_binary, lgdchis_binary, count_mat_binary = dm.gridcount2D(bm0_act,signlog(bchi_s_act),mag_bins,lgdchis_bins)
    mag_binary_true, lgdchis_binary_true, count_mat_binary_true = dm.gridcount2D(bm0_act_true,signlog(bchi_s_act_true),mag_bins,lgdchis_bins)
    
    

    print(np.sum(count_mat_binary))
    print(np.sum(count_mat_binary_true))


    count_mat_binary = np.abs(count_mat_binary-0.5)+0.5
    rate_mat_binary = count_mat_binary_true/count_mat_binary
    rate_mat_binary = (rate_mat_binary//0.25)/4

    plt.figure(figsize=(12,12))
    plt.yticks(range(len(mag_binary)))
    plt.gca().set_yticklabels(mag_binary)
    plt.xticks(range(len(lgdchis_binary)))
    plt.gca().set_xticklabels(lgdchis_binary)
    plt.imshow(rate_mat_binary, cmap=plt.cm.hot_r)
    # plt.colorbar()
    plt.xlabel("$\log_{10} \Delta \chi^2$(single fitting)")
    plt.ylabel("$m_0$")
    plt.title("range of $\log_{10}q$: (-%s.%s, -%s.%s )"%(name_group[-2],name_group[-1],name_group[0],name_group[1]))
    plt.savefig("test_binary_rate_bins_"+name_group+".png")
    plt.close()

    line_posi = dm.getborder(rate_mat_binary)

    data_hist = np.array([list(mag_binary), list(lgdchis_binary), list(rate_mat_binary), list(line_posi)],dtype=object)

    np.save("data_hist_"+name_group+".npy",data_hist,allow_pickle=True)

def drawROCcurve(datadir=rootval):
    # initialize model
    network = lcnet.ResNet()
    criterion = nn.BCELoss()
    if use_gpu:
        # network = network.cuda()
        criterion = criterion.cuda()
        network = nn.DataParallel(network).cuda()

    network.load_state_dict(torch.load(path_params+preload_Netmodel))

    label0_pre = []
    label1_pre = []

    # file_actual/predicted
    file_bb = []
    file_bs = []
    file_sb = []
    file_ss = []

    num_batch = num_test//batch_test

    if num_test%batch_test != 0:
        num_batch += 1

    # making prediction
    for index_batch in range(num_batch):

        input_batch = []
        label_batch = []
        chi_s_batch = []
        file_batch = []
        m0_batch = []


        for index in range(index_batch*batch_test,np.min([(index_batch+1)*batch_test,num_test])):
            lc_data,args = lcnet.default_loader_fortest(data_root = datadir,posi_lc = index)
            # [u_0, rho, q, s, alpha, t_E, basis_m, t_0, dchis, label]
            input_batch.append(lc_data)

            file_batch.append(index)

            dchi_s = args[-2]
            bslabel = args[-1]
            m0 = args[6]

            
            if (bslabel>0.5)&(dchi_s < 10):
                bslabel = 0
            
            label_batch.append(bslabel)
            chi_s_batch.append(dchi_s)
            m0_batch.append(m0)
        
        input_batch = torch.from_numpy(np.array(input_batch)).float()
        if use_gpu:
            input_batch = input_batch.cuda()
        
        network.eval()
        output_batch = network(input_batch).detach().cpu().numpy()


        bspre_batch = output_batch.T[0]
        label_batch = np.array(label_batch)

        label0_pre_batch = bspre_batch[label_batch<0.5]
        label1_pre_batch = bspre_batch[label_batch>0.5]

        if index_batch == 0:
            label0_pre = label0_pre_batch.copy()
            label1_pre = label1_pre_batch.copy()

        else:
            label0_pre = np.append(label0_pre,label0_pre_batch)
            label1_pre = np.append(label1_pre,label1_pre_batch)

    label0_pre = np.array(label0_pre)
    label1_pre = np.array(label1_pre)

    threshold_list = np.linspace(0,1,100)
    Tp_list = []
    Fp_list = []
    Tn_list = []
    Fn_list = []

    for threshold in threshold_list:
        Tp = np.sum(label1_pre>=threshold)
        Fp = np.sum(label0_pre>threshold)
        Tn = np.sum(label0_pre<=threshold)
        Fn = np.sum(label1_pre<threshold)
        '''
        Tpr = np.sum(label1_pre>=threshold)/len(label1_pre)
        Fpr = np.sum(label0_pre>threshold)/len(label0_pre)
        Tnr = np.sum(label0_pre<=threshold)/len(label0_pre)
        Fnr = np.sum(label1_pre<threshold)/len(label1_pre)
        '''
        Tp_list.append(Tp)
        Fp_list.append(Fp)
        Tn_list.append(Tn)
        Fn_list.append(Fn)

    
    plt.figure()
    plt.plot(np.array(Fp_list)/(np.array(Fp_list)+np.array(Tn_list)),np.array(Tp_list)/(np.array(Tp_list)+np.array(Fn_list)))
    plt.plot(np.array(Tp_list)/(np.array(Tp_list)+np.array(Fn_list)),np.array(Tp_list)/(np.array(Tp_list)+np.array(Fp_list)))
    plt.plot([0,1],[0,1],linestyle="--")
    plt.xlabel("FPR/recall")
    plt.ylabel("TPR/precision")
    plt.axis("scaled")
    plt.title("range of $\log_{10}q$: (-%s.%s, -%s.%s )"%(name_group[-2],name_group[-1],name_group[0],name_group[1]))
    plt.savefig("ROC_PR_"+name_group+".png")
    plt.close()

    np.save("rate_TFpn"+name_group+".npy",np.array([Tp_list,Fp_list,Tn_list,Fn_list]))
    

if __name__=="__main__":
    drawROCcurve()

    
