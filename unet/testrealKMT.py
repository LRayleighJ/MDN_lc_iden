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
import datamodule.loaddata as loaddata
import netmodule.unetforkmt as lcnet


np.set_printoptions(suppress=True)

preload_Netmodel = "GRU_unet.pkl"
path_params = "/scratch/zerui603/netparams/"
paramsid = 40

use_gpu = torch.cuda.is_available()
print("GPU:", use_gpu)

def testdraw2019(posi):
    args,data_total,data_A,data_C,data_S,_ = loaddata.getKMTdata(posi=posi,cutratio=[-2,2],FWHM_threshold=50,sky_threshold=20000)
    # args_fit,data_total_fit,data_A_fit,data_C_fit,data_S_fit,_ = loaddata.getKMTdata(posi=posi,cutratio=[-10,10],FWHM_threshold=50,sky_threshold=20000)
    print(posi,args.astype(np.float))
    time,lc,sig = data_total
    time_A,lc_A,sig_A = data_A
    time_C,lc_C,sig_C = data_C
    time_S,lc_S,sig_S = data_S

    time_A_fit,lc_A_fit,sig_A_fit = data_A_fit
    time_C_fit,lc_C_fit,sig_C_fit = data_C_fit
    time_S_fit,lc_S_fit,sig_S_fit = data_S_fit

    '''
    args_single_fit_A = loaddata.fitKMTdata(data_A_fit,args_fit)
    args_single_fit_C = loaddata.fitKMTdata(data_C_fit,args_fit)
    args_single_fit_S = loaddata.fitKMTdata(data_S_fit,args_fit)
    lc_A_single = loaddata.KMTmag_fit(loaddata.get_test_time_series(time_A_fit),*args_single_fit_A)
    lc_C_single = loaddata.KMTmag_fit(loaddata.get_test_time_series(time_C_fit),*args_single_fit_C)
    lc_S_single = loaddata.KMTmag_fit(loaddata.get_test_time_series(time_S_fit),*args_single_fit_S)
    '''
    # time_align, mag_align, err_align = loaddata.alignKMTdata([data_A,data_C,data_S],args)
    
    plt.figure(figsize=(10,5))
    plt.errorbar(time_A,lc_A,yerr=sig_A,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    plt.errorbar(time_C,lc_C,yerr=sig_C,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    plt.errorbar(time_S,lc_S,yerr=sig_S,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    # plt.errorbar(time_A_fit,lc_A_fit,yerr=sig_A_fit,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    # plt.plot(loaddata.get_test_time_series(time_A_fit),lc_A_single,ls="--")
    # plt.errorbar(time_C_fit,lc_C_fit,yerr=sig_C_fit,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    # plt.plot(loaddata.get_test_time_series(time_C_fit),lc_C_single,ls="--")
    # plt.errorbar(time_S_fit,lc_S_fit,yerr=sig_S_fit,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    # plt.plot(loaddata.get_test_time_series(time_S_fit),lc_S_single,ls="--")
    # plt.errorbar(time_align,mag_align,yerr=err_align,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    # plt.scatter(time,lc,s=4,c="r",label = "lightcurve")
    plt.xlabel("time/HJD")
    plt.ylabel("magnitude")
    # plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/realfig/%04d.pdf"%(posi,))
    plt.close()

    return len(time)


def static_KMT(posi):
    print("index: ", posi)
    KMT_args,data_total,data_A,data_C,data_S,_ = loaddata.getKMTdata(posi=posi,cutratio=[-3,3],FWHM_threshold=50,sky_threshold=20000)
    time,mag,err = data_total

    minimize_args, mag_singlemodel = loaddata.doublefitting(time,mag,err,KMT_args)
    time1 = time.copy()

    new_cutratio = minimize_args[0]/KMT_args[0]
    new_correctratio = (minimize_args[1]-KMT_args[1])/KMT_args[0]

    print("t0: ",minimize_args[1],KMT_args[1])
    
    '''
    plt.figure(figsize=(10,5))
    plt.errorbar(time,mag,yerr=err,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    plt.plot(time,mag_singlemodel,ls="--")
    plt.xlabel("time/HJD")
    plt.ylabel("magnitude")
    # plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/realfig/%04d.png"%(posi,))
    plt.close()
    '''

    KMT_args,data_total,data_A,data_C,data_S,_ = loaddata.getKMTdata(posi=posi,cutratio=[-2*new_cutratio+new_correctratio,2*new_cutratio+new_correctratio],FWHM_threshold=50,sky_threshold=20000)
    time,mag,err = data_total
    print("cut_ratio: ", new_cutratio)
    print("size: ", len(time))

    if len(time) < 1000:
        return 

    plt.figure(figsize=(10,5))
    plt.errorbar(time,mag,yerr=err,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    plt.plot(time1,mag_singlemodel,ls="--")
    plt.xlabel("time/HJD")
    plt.ylabel("magnitude")
    plt.gca().invert_yaxis()
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/realfig/%04d_compare.png"%(posi,))
    plt.close()


def testUnet_KMT2019(posi, network=None):
    print("index: ", posi)
    KMT_args,data_total,data_A,data_C,data_S,_ = loaddata.getKMTdata(posi=posi,cutratio=[-3,3],FWHM_threshold=50,sky_threshold=20000)
    time,mag,err = data_total

    minimize_args, mag_singlemodel = loaddata.doublefitting(time,mag,err,KMT_args)

    new_cutratio = minimize_args[0]/KMT_args[0]
    new_correctratio = (minimize_args[1]-KMT_args[1])/KMT_args[0]

    print("t0: ",minimize_args[1],KMT_args[1])

    KMT_args,data_total,data_A,data_C,data_S,_ = loaddata.getKMTdata(posi=posi,cutratio=[-2*new_cutratio+new_correctratio,2*new_cutratio+new_correctratio],FWHM_threshold=50,sky_threshold=20000)
    time,mag,err = data_total
    print("cut_ratio: ", new_cutratio)
    print("size: ", len(time))

    if len(time) < 1000:
        return 

    time_rs,mag_rs,err_rs = lcnet.sample_curve(time,mag,err)
    mag_single_rs = loaddata.mag_cal(time_rs,*minimize_args)

    data_input = np.array(lcnet.loader_transform(time_rs,mag_rs,err_rs))

    data_input = torch.from_numpy(data_input).float()
    if use_gpu:
        data_input = data_input.cuda()
    
    network.eval()
    data_output = network(data_input).detach().cpu().numpy()

    print("Output of network: ", data_output.shape)

    bspre_array = data_output[0][1]

    thres = 0.998

    plt.figure(figsize=(10,5))
    plt.errorbar(time_rs[bspre_array<thres],mag_rs[bspre_array<thres],yerr=err_rs[bspre_array<thres],fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,c="blue",label="single")
    plt.errorbar(time_rs[bspre_array>thres],mag_rs[bspre_array>thres],yerr=err_rs[bspre_array>thres],fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,c="red", label="binary")
    plt.plot(time_rs,mag_single_rs,ls="--")
    plt.xlabel("time/HJD")
    plt.ylabel("magnitude")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/realfig/%04d_compare.png"%(posi,))
    plt.close()




if __name__=="__main__":

    KMT2019anomaly = np.loadtxt("/home/zerui603/MDN_lc_iden/unet/KMT2019anomaly.txt").astype(np.int64)

    network = lcnet.Unet()
    if use_gpu:
        network = nn.DataParallel(network).cuda()
    
    network.load_state_dict(torch.load(path_params+preload_Netmodel[:-4]+"_"+str(paramsid)+".pkl"))

    for posi in KMT2019anomaly:
        try:
            testUnet_KMT2019(posi,network)
        except:
            continue


    '''
    len_list = []
    
    for i in range(1,500):
        try:
            length_lc = testdraw2019(i)
            len_list.append(length_lc)
        except:
            print(i,"error")
            print(traceback.format_exc())
            continue
    plt.figure()
    plt.hist(np.log10(len_list),bins=50)
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/realKMT_distri.png")
    plt.close()
    '''