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

name_group_test_list = ["00to10test","10to20test","20to30test","30to40test"]# ["00to05test","05to10test","10to15test","15to20test","20to25test","25to30test","30to35test","35to40test"]
move_target_list = ["00to10test","10to20test","20to30test","30to40test"]

use_gpu = torch.cuda.is_available()

move_index = 0

def move_file(index):
    global move_index
    rootdir1 = "/scratch/zerui603/KMT_simu_lowratio/qseries/"+name_group_test_list[2*move_index]+"/"
    rootdir2 = "/scratch/zerui603/KMT_simu_lowratio/qseries/"+name_group_test_list[2*move_index+1]+"/"
    targetdir = "/scratch/zerui603/KMT_simu_lowratio/qseries/"+move_target_list[move_index]+"/"
    command1 = "cp "+rootdir1+str(index)+".npy "+targetdir+str(index)+".npy"
    command2 = "cp "+rootdir2+str(index)+".npy "+targetdir+str(index+100000)+".npy"
    os.system(command1)
    os.system(command2)

def signlog(x):
    return np.log10(np.abs(x))


def testnet(name_group="",network=None):
    num_test = 200000
    batch_test = 15000
    rootval = "/scratch/zerui603/KMT_simu_lowratio/qseries/"+name_group+"/"
    rootdraw = "/home/zerui603/MDN_lc/iden_1D/testfig/"
    fullrootdraw = "/scratch/zerui603/figtest_iden/"+name_group+"/"
    # mag_bins = np.linspace(16,20,17)
    # lgdchis_bins = np.linspace(1.5,3.5,17)
    # initialize model
    
    
    bspre_total = []
    label_total = []
    dchis_total = []

    D2_total = []
    m0_total = []
    
    ID_fortest = []
    label_fortest = []
    bspre_fortest = []

    num_batch = num_test//batch_test

    if num_test%batch_test != 0:
        num_batch += 1

    # making prediction
    for index_batch in range(num_batch):

        input_batch = []
        label_batch = []
        dchis_batch = []
        D2_batch = []
        m0_batch = []

        # pre_total_batch = []
        # chis_total_batch = []


        for index in range(index_batch*batch_test,np.min([(index_batch+1)*batch_test,num_test])):
            lc_data,args,data_extra = lcnet.default_loader_fortest(data_root = rootval,posi_lc = index,extra_index=[1,5,7])
            # [u_0, rho, q, s, alpha, t_E, basis_m, t_0, dchis, label]
            input_batch.append(lc_data)

            dchis = args[-2]
            bslabel = args[-1]
            m0 = args[6]

            # test ID

            time,lc_mb,lc_ms = data_extra
            D2 = np.sum((lc_mb-lc_ms)**2)

            if (bslabel>0.5)&(dchis < 10):
                bslabel = 0

            if (dchis>10**1)&(dchis<10**3):
                ID_fortest.append(index)
                label_fortest.append(bslabel)
            
            label_batch.append(bslabel)
            dchis_batch.append(dchis)
            D2_batch.append(D2)
            m0_batch.append(m0)
        
        input_batch = torch.from_numpy(np.array(input_batch)).float()
        if use_gpu:
            input_batch = input_batch.cuda()
        
        network.eval()
        output_batch = network(input_batch).detach().cpu().numpy()

        bspre_batch = output_batch.T[0]
        label_batch = np.array(label_batch).astype(np.int)

        
        dchis_batch = np.array(dchis_batch)
        bspre_fortest = np.append(bspre_fortest,bspre_batch[(dchis_batch<10**3)&(dchis_batch>10**1)])

        bspre_total = np.append(bspre_total,bspre_batch)
        label_total = np.append(label_total,label_batch)
        dchis_total = np.append(dchis_total,dchis_batch)
        D2_total = np.append(D2_total,D2_batch)
        m0_total = np.append(m0_total,m0_batch)
    
    np.save("mixdatanet_"+name_group+".npy",np.array([bspre_total,label_total,dchis_total]))

    thres_list = np.linspace(0,0.99,100)
    threshold=0

    for thres in thres_list:
        FPrate = np.sum((bspre_total>thres)*(np.abs(dchis_total)<10))/np.sum(bspre_total>thres)
        if FPrate<0.01:
            print(name_group, thres)
            threshold = thres
            break

    np.save("eyetest_index_"+name_group+".npy", np.array([ID_fortest, label_fortest, bspre_fortest]))


    # hist
    lgdchis_total = np.log10(np.abs(dchis_total))

    # count 

    lgdchis_pa11 = lgdchis_total[(bspre_total>=threshold)&(label_total>0.5)]
    lgdchis_pa10 = lgdchis_total[(bspre_total>=threshold)&(label_total<0.5)]
    lgdchis_pa01 = lgdchis_total[(bspre_total<threshold)&(label_total>0.5)]
    lgdchis_pa00 = lgdchis_total[(bspre_total<threshold)&(label_total<0.5)]

    print("pa11,pa10,pa01,pa00: ",len(lgdchis_pa11),len(lgdchis_pa10),len(lgdchis_pa01),len(lgdchis_pa00))

    lgdchis_act0 = lgdchis_total[label_total<0.5]
    lgdchis_act1 = lgdchis_total[label_total>0.5]
    lgdchis_pre0 = lgdchis_total[bspre_total<threshold]
    lgdchis_pre1 = lgdchis_total[bspre_total>=threshold]

    plt.figure(figsize=[7,6])

    plt.hist(lgdchis_act1,bins=80,range=(-1,7),label="actual: binary",histtype="step",ls="--",color="blue")
    plt.hist(lgdchis_act0,bins=80,range=(-1,7),label="actual: single",histtype="step",ls="--",color="red")
    plt.hist(lgdchis_pre1,bins=80,range=(-1,7),label="predict: binary",histtype="step",color="blue")
    plt.hist(lgdchis_pre0,bins=80,range=(-1,7),label="predict: single",histtype="step",color="red")

    plt.xlabel("$\log_{10} |\Delta \chi^2|$",fontsize=20)
    plt.tick_params(labelsize=15)
    plt.legend()
    # plt.title("range of $\log_{10}q$: (-%s.%s, -%s.%s )"%(name_group[-2],name_group[-1],name_group[0],name_group[1]),fontsize=30)
    plt.savefig("/home/zerui603/MDN_lc_iden/iden_1D/histbs_mixdatanet_"+name_group+".pdf")
    plt.close()

    hist_act0,_ = np.histogram(lgdchis_act0,bins=80,range=(-1,7))
    hist_act1,_ = np.histogram(lgdchis_act1,bins=80,range=(-1,7))
    hist_pre0,_ = np.histogram(lgdchis_pre0,bins=80,range=(-1,7))
    hist_pre1,_ = np.histogram(lgdchis_pre1,bins=80,range=(-1,7))


    ## test structure
    D2s_total = D2_total[label_total<0.5]
    D2b_total = D2_total[label_total>0.5]
    m0s_total = m0_total[label_total<0.5]
    m0b_total = m0_total[label_total>0.5]

    plt.figure(figsize=(9,9))
    # plt.scatter(m0s_total,np.log10(D2s_total),s=1.5,alpha=0.5,label="$\log_{10}D^2$ (single events)")
    plt.scatter(m0b_total,np.log10(D2b_total),s=1.5,alpha=0.5,label="$\log_{10}D^2$ (binary events)")
    plt.xlim(16.1,19.9)
    plt.xlabel("$m_0$")
    plt.ylabel("$\log_{10} D^2$")
    plt.savefig("/home/zerui603/MDN_lc_iden/iden_1D/structurem0_mixdatanet_"+name_group+".pdf")
    plt.close()


    # test dchis-m0 bins distribution

    # bspre_total = []
    # label_total = []
    # dchis_total = []
    # D2_total = []
    # m0_total = []

    mag_bins = np.linspace(16,20,26)

    lgdchis_bins = np.ones(mag_bins.shape)

    bchi_s_act = dchis_total[label_total>0.5]
    bchi_s_act_forpercent = bchi_s_act.copy()
    bchi_s_act_true = dchis_total[(label_total>0.5)&(bspre_total>threshold)]

    lgbchi_s_act_sort = np.sort(signlog(bchi_s_act))

    bm0_act = m0_total[label_total>0.5]
    bm0_act_true = m0_total[(label_total>0.5)&(bspre_total>threshold)]

    rate_testhistbins = 0.9

    for i in range(len(lgdchis_bins)):
        delta = np.int(rate_testhistbins*len(lgbchi_s_act_sort)/len(lgdchis_bins))
        lgdchis_bins[i] = lgbchi_s_act_sort[i*delta]
    
    lgdchis_bins[0] = 1
    lgdchis_bins[-1] = lgbchi_s_act_sort[-1]


    mag_binary, lgdchis_binary, count_mat_binary = dm.gridcount2D(bm0_act,signlog(bchi_s_act),mag_bins,lgdchis_bins)
    mag_binary_true, lgdchis_binary_true, count_mat_binary_true = dm.gridcount2D(bm0_act_true,signlog(bchi_s_act_true),mag_bins,lgdchis_bins)
    

    count_mat_binary = np.abs(count_mat_binary-0.5)+0.5
    rate_mat_binary = count_mat_binary_true/count_mat_binary
    rate_mat_binary = (rate_mat_binary//0.25)/4
    rate_mat_binary[-1][0] = 0
    rate_mat_binary[0][-1] = 1

    plt.figure(figsize=(12,13))
    plt.yticks(range(len(mag_binary)),size=20)
    m0label = ["%.2f"%(x,) for x in mag_binary]
    plt.gca().set_yticklabels(m0label)
    plt.xticks(range(len(lgdchis_binary)),size=20)
    plt.tick_params(labelsize=20)
    chilabel = ["%.2f%%"%x for x in np.linspace(0,100*rate_testhistbins,len(lgdchis_binary))]
    plt.gca().set_xticklabels(chilabel, rotation=45)
    plt.imshow(rate_mat_binary, cmap=plt.cm.hot_r)
    # plt.colorbar()
    plt.xlabel("$\log_{10} \Delta \chi^2$(single fitting)",fontsize=30)
    plt.ylabel("$m_0$",fontsize=30)
    # plt.title("range of $\log_{10}q$: (-%s.%s, -%s.%s )"%(name_group[-2],name_group[-1],name_group[0],name_group[1]))
    plt.savefig("/home/zerui603/MDN_lc_iden/iden_1D/test_binary_rate_percent_"+name_group+".pdf")
    plt.close()

    # test test dchis-m0 bins distribution

    mag_bins = np.linspace(16,20,26)

    lgdchis_bins = np.linspace(1.5,3,26)

    bchi_s_act = dchis_total[label_total>0.5]
    bchi_s_act_true = dchis_total[(label_total>0.5)&(bspre_total>threshold)]

    lgbchi_s_act_sort = np.sort(signlog(bchi_s_act))

    bm0_act = m0_total[label_total>0.5]
    bm0_act_true = m0_total[(label_total>0.5)&(bspre_total>threshold)]


    mag_binary, lgdchis_binary, count_mat_binary = dm.gridcount2D(bm0_act,signlog(bchi_s_act),mag_bins,lgdchis_bins)
    mag_binary_true, lgdchis_binary_true, count_mat_binary_true = dm.gridcount2D(bm0_act_true,signlog(bchi_s_act_true),mag_bins,lgdchis_bins)
    

    count_mat_binary = np.abs(count_mat_binary-0.5)+0.5
    rate_mat_binary = count_mat_binary_true/count_mat_binary
    rate_mat_binary = (rate_mat_binary//0.25)/4
    rate_mat_binary[-1][0] = 0
    rate_mat_binary[0][-1] = 1

    plt.figure(figsize=(12,13))
    plt.yticks(range(len(mag_binary)),size=20)
    plt.xticks(range(len(lgdchis_binary)),size=20)
    m0label = ["%.2f"%(x,) for x in mag_binary]
    plt.tick_params(labelsize=20)
    plt.gca().set_yticklabels(m0label)
    xlabel = ["%.2f"%x for x in lgdchis_binary]
    plt.gca().set_xticklabels(xlabel, rotation=45)
    plt.imshow(rate_mat_binary, cmap=plt.cm.hot_r)
    # plt.colorbar()
    plt.xlabel("$\log_{10} \Delta \chi^2$(single fitting)",fontsize=30)
    plt.ylabel("$m_0$",fontsize=30)
    # plt.title("range of $\log_{10}q$: (-%s.%s, -%s.%s )"%(name_group[-2],name_group[-1],name_group[0],name_group[1]))
    plt.savefig("/home/zerui603/MDN_lc_iden/iden_1D/test_binary_rate_dchis_"+name_group+".pdf")
    plt.close()

    # test D2 rate

    mag_bins = np.linspace(16,20,26)

    lgdchis_bins = np.linspace(-3,2,26)

    bchi_s_act = D2_total[label_total>0.5]
    bchi_s_act_true = D2_total[(label_total>0.5)&(bspre_total>threshold)]

    lgbchi_s_act_sort = np.sort(signlog(bchi_s_act))

    bm0_act = m0_total[label_total>0.5]
    bm0_act_true = m0_total[(label_total>0.5)&(bspre_total>threshold)]


    mag_binary, lgdchis_binary, count_mat_binary = dm.gridcount2D(bm0_act,signlog(bchi_s_act),mag_bins,lgdchis_bins)
    mag_binary_true, lgdchis_binary_true, count_mat_binary_true = dm.gridcount2D(bm0_act_true,signlog(bchi_s_act_true),mag_bins,lgdchis_bins)
    

    count_mat_binary = np.abs(count_mat_binary-0.5)+0.5
    rate_mat_binary = count_mat_binary_true/count_mat_binary
    rate_mat_binary = (rate_mat_binary//0.25)/4
    rate_mat_binary[-1][0] = 0
    rate_mat_binary[0][-1] = 1

    plt.figure(figsize=(12,13))
    plt.yticks(range(len(mag_binary)),size=20)
    plt.xticks(range(len(lgdchis_binary)),size=20)
    plt.tick_params(labelsize=20)
    m0label = ["%.2f"%(x,) for x in mag_binary]
    plt.gca().set_yticklabels(m0label)
    print(lgdchis_binary)
    xlabel = ["%.2f"%x for x in lgdchis_binary]
    plt.gca().set_xticklabels(xlabel, rotation=45)
    plt.imshow(rate_mat_binary, cmap=plt.cm.hot_r)
    # plt.colorbar()
    plt.xlabel("$\log_{10} D^2$ (single fitting)",fontsize=30)
    plt.ylabel("$m_0$",fontsize=30)
    # plt.title("range of $\log_{10}q$: (-%s.%s, -%s.%s )"%(name_group[-2],name_group[-1],name_group[0],name_group[1]))
    plt.savefig("/home/zerui603/MDN_lc_iden/iden_1D/test_binary_rate_D2_"+name_group+".pdf")
    plt.close()


    # percent_count

    rate_testhistbins = 0.75

    hist_xaxis_percent = np.ones(101)
    lgbchi_s_act_sort = np.sort(signlog(bchi_s_act_forpercent))

    for i in range(101):
        delta = np.int(rate_testhistbins*len(lgbchi_s_act_sort)/len(hist_xaxis_percent))
        hist_xaxis_percent[i] = lgbchi_s_act_sort[i*delta]
    
    hist_act0_percent,_ = np.histogram(lgdchis_act0,bins=hist_xaxis_percent)
    hist_act1_percent,_ = np.histogram(lgdchis_act1,bins=hist_xaxis_percent)
    hist_pre0_percent,_ = np.histogram(lgdchis_pre0,bins=hist_xaxis_percent)
    hist_pre1_percent,_ = np.histogram(lgdchis_pre1,bins=hist_xaxis_percent)

    print(hist_xaxis_percent)

    return np.array([hist_act0,hist_act1,hist_pre0,hist_pre1]),np.array([hist_act0_percent,hist_act1_percent,hist_pre0_percent,hist_pre1_percent])

if __name__ == "__main__":
    '''
    for move_index in range(4):
        with mp.Pool(20) as p:
            p.map(move_file, range(100000))
    '''
    network = lcnet.ResNet()
    if use_gpu:
        network = nn.DataParallel(network).cuda()
    preload_Netmodel = "GRUresnet_iden_res_mix.pkl"
    path_params = "/scratch/zerui603/netparams/"
    network.load_state_dict(torch.load(path_params+preload_Netmodel))

    ratedata_list = []
    ratedata_percent_list = []
    
    for i,name_group in enumerate(name_group_test_list):
        hist_data,hist_data_percent = testnet(name_group,network)
        ratedata_list.append(hist_data)
        ratedata_percent_list.append(hist_data_percent)
        # labeldata_list.append("range of $\log_{10}q$: (-%s.%s, -%s.%s )"%(name_group[4],name_group[5],name_group[0],name_group[1]))
    np.save("ratedata_mixnet.npy",ratedata_list)
    np.save("ratedata_mixnet_percent.npy",ratedata_percent_list)
    

    labeldata_list = []
    for i,name_group in enumerate(name_group_test_list):
        labeldata_list.append("$\log_{10}q\sim$(-%s.%s, -%s.%s )"%(name_group[4],name_group[5],name_group[0],name_group[1]))


    # ratedata_list = np.load("ratedata_mixnet.npy",allow_pickle=True)

    print("check 1")

    hist_axis = np.linspace(-1,7,80)
    plt.figure(figsize=(10,4))
    for i,ratedata in enumerate(ratedata_list):
        
        hist_lgdchis_act0,hist_lgdchis_act1,hist_lgdchis_pre0,hist_lgdchis_pre1 = ratedata

        print(ratedata.shape)

        hist_lgdchis_act0 = np.array(hist_lgdchis_act0)
        hist_lgdchis_act1 = np.array(hist_lgdchis_act1)
        hist_lgdchis_pre0 = np.array(hist_lgdchis_pre0)
        hist_lgdchis_pre1 = np.array(hist_lgdchis_pre1)

        hist_lgdchis_act_total = hist_lgdchis_act0+hist_lgdchis_act1

        rate_line_0 = hist_lgdchis_pre0[hist_lgdchis_act_total>0]/hist_lgdchis_act_total[hist_lgdchis_act_total>0]
        rate_line_1 = hist_lgdchis_pre1[hist_lgdchis_act_total>0]/hist_lgdchis_act_total[hist_lgdchis_act_total>0]
        hist_axis_temp0 = hist_axis[hist_lgdchis_act_total>0]
        hist_axis_temp1 = hist_axis[hist_lgdchis_act_total>0]
        # plt.plot(hist_axis_temp0, rate_line_0)
        plt.plot(hist_axis_temp1, rate_line_1,label=labeldata_list[i])
    plt.xlabel("$\log_{10} |\Delta \chi^2|$",fontsize=25)
    plt.ylabel("$Accuracy$",fontsize=25)
    plt.tick_params(labelsize=20)
    plt.xlim((1,4))
    plt.legend(fontsize=15)
    plt.savefig("/home/zerui603/MDN_lc_iden/iden_1D/ratehistb_mixdatanet.pdf")
    plt.close()


    plt.figure(figsize=(10,4))
    for i,ratedata in enumerate(ratedata_list):
        
        hist_lgdchis_act0,hist_lgdchis_act1,hist_lgdchis_pre0,hist_lgdchis_pre1 = ratedata

        print(ratedata.shape)

        hist_lgdchis_act0 = np.array(hist_lgdchis_act0)
        hist_lgdchis_act1 = np.array(hist_lgdchis_act1)
        hist_lgdchis_pre0 = np.array(hist_lgdchis_pre0)
        hist_lgdchis_pre1 = np.array(hist_lgdchis_pre1)

        hist_lgdchis_act_total = hist_lgdchis_act0+hist_lgdchis_act1

        rate_line_0 = hist_lgdchis_pre0[hist_lgdchis_act_total>0]/hist_lgdchis_act_total[hist_lgdchis_act_total>0]
        rate_line_1 = hist_lgdchis_pre1[hist_lgdchis_act_total>0]/hist_lgdchis_act_total[hist_lgdchis_act_total>0]
        hist_axis_temp0 = hist_axis[hist_lgdchis_act_total>0]
        hist_axis_temp1 = hist_axis[hist_lgdchis_act_total>0]
        plt.plot(hist_axis_temp0, rate_line_0,label=labeldata_list[i])
        # plt.plot(hist_axis_temp1, rate_line_1,label=labeldata_list[i])
    plt.xlabel("$\log_{10} |\Delta \chi^2|$",fontsize=25)
    plt.ylabel("$Accuracy$",fontsize=25)
    plt.tick_params(labelsize=20)
    plt.xlim((0,4))
    plt.legend(fontsize=15)
    plt.savefig("/home/zerui603/MDN_lc_iden/iden_1D/ratehists_mixdatanet.pdf")
    plt.close()

    print("check 2")

    hist_axis = np.linspace(0,75,100)
    plt.figure(figsize=(10,4))
    for i,ratedata in enumerate(ratedata_percent_list):
        
        hist_lgdchis_act0,hist_lgdchis_act1,hist_lgdchis_pre0,hist_lgdchis_pre1 = ratedata

        print(ratedata.shape)

        hist_lgdchis_act0 = np.array(hist_lgdchis_act0)
        hist_lgdchis_act1 = np.array(hist_lgdchis_act1)
        hist_lgdchis_pre0 = np.array(hist_lgdchis_pre0)
        hist_lgdchis_pre1 = np.array(hist_lgdchis_pre1)

        hist_lgdchis_act_total = hist_lgdchis_act0+hist_lgdchis_act1

        rate_line_0 = hist_lgdchis_pre0[hist_lgdchis_act_total>0]/hist_lgdchis_act_total[hist_lgdchis_act_total>0]
        rate_line_1 = hist_lgdchis_pre1[hist_lgdchis_act_total>0]/hist_lgdchis_act_total[hist_lgdchis_act_total>0]
        hist_axis_temp0 = hist_axis[hist_lgdchis_act_total>0]
        hist_axis_temp1 = hist_axis[hist_lgdchis_act_total>0]
        # plt.plot(hist_axis_temp0, rate_line_0)
        plt.plot(hist_axis_temp1, rate_line_1,label=labeldata_list[i])
    plt.xlabel("$percent$",fontsize=25)
    plt.ylabel("$Accuracy$",fontsize=25)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=15)
    plt.savefig("/home/zerui603/MDN_lc_iden/iden_1D/ratepercentb_mixdatanet.pdf")
    plt.close()


    plt.figure(figsize=(10,4))
    for i,ratedata in enumerate(ratedata_percent_list):
        
        hist_lgdchis_act0,hist_lgdchis_act1,hist_lgdchis_pre0,hist_lgdchis_pre1 = ratedata

        print(ratedata.shape)

        hist_lgdchis_act0 = np.array(hist_lgdchis_act0)
        hist_lgdchis_act1 = np.array(hist_lgdchis_act1)
        hist_lgdchis_pre0 = np.array(hist_lgdchis_pre0)
        hist_lgdchis_pre1 = np.array(hist_lgdchis_pre1)

        hist_lgdchis_act_total = hist_lgdchis_act0+hist_lgdchis_act1

        rate_line_0 = hist_lgdchis_pre0[hist_lgdchis_act_total>0]/hist_lgdchis_act_total[hist_lgdchis_act_total>0]
        rate_line_1 = hist_lgdchis_pre1[hist_lgdchis_act_total>0]/hist_lgdchis_act_total[hist_lgdchis_act_total>0]
        hist_axis_temp0 = hist_axis[hist_lgdchis_act_total>0]
        hist_axis_temp1 = hist_axis[hist_lgdchis_act_total>0]
        plt.plot(hist_axis_temp0, rate_line_0,label=labeldata_list[i])
        # plt.plot(hist_axis_temp1, rate_line_1,label=labeldata_list[i])
    plt.xlabel("$percent$",fontsize=25)
    plt.ylabel("$Accuracy$",fontsize=25)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=15)
    plt.savefig("/home/zerui603/MDN_lc_iden/iden_1D/ratepercents_mixdatanet.pdf")
    plt.close()
    

        

        
    