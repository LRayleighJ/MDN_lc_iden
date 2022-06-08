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
import traceback

import datamodule.dm as dm
import datamodule.loaddata as loaddata
import netmodule.unetforkmt as lcnet

np.set_printoptions(suppress=True)

def testdraw2019(posi):
    args,data_total,data_A,data_C,data_S,_ = loaddata.getKMTdata(posi=posi,cutratio=[-1,1],FWHM_threshold=50,sky_threshold=20000)
    args_fit,data_total_fit,data_A_fit,data_C_fit,data_S_fit,_ = loaddata.getKMTdata(posi=posi,cutratio=[-10,10],FWHM_threshold=50,sky_threshold=20000)
    print(posi,args.astype(np.float))
    time,lc,sig = data_total
    time_A,lc_A,sig_A = data_A
    time_C,lc_C,sig_C = data_C
    time_S,lc_S,sig_S = data_S

    time_A_fit,lc_A_fit,sig_A_fit = data_A_fit
    time_C_fit,lc_C_fit,sig_C_fit = data_C_fit
    time_S_fit,lc_S_fit,sig_S_fit = data_S_fit


    args_single_fit_A = loaddata.fitKMTdata(data_A_fit,args_fit)
    args_single_fit_C = loaddata.fitKMTdata(data_C_fit,args_fit)
    args_single_fit_S = loaddata.fitKMTdata(data_S_fit,args_fit)
    lc_A_single = loaddata.KMTmag_fit(loaddata.get_test_time_series(time_A_fit),*args_single_fit_A)
    lc_C_single = loaddata.KMTmag_fit(loaddata.get_test_time_series(time_C_fit),*args_single_fit_C)
    lc_S_single = loaddata.KMTmag_fit(loaddata.get_test_time_series(time_S_fit),*args_single_fit_S)
    
    # time_align, mag_align, err_align = loaddata.alignKMTdata([data_A,data_C,data_S],args)
    
    plt.figure(figsize=(10,5))
    # plt.errorbar(time_C,lc_C,yerr=sig_C,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    plt.errorbar(time_A_fit,lc_A_fit,yerr=sig_A_fit,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    plt.plot(loaddata.get_test_time_series(time_A_fit),lc_A_single,ls="--")

    plt.errorbar(time_C_fit,lc_C_fit,yerr=sig_C_fit,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    plt.plot(loaddata.get_test_time_series(time_C_fit),lc_C_single,ls="--")

    plt.errorbar(time_S_fit,lc_S_fit,yerr=sig_S_fit,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    plt.plot(loaddata.get_test_time_series(time_S_fit),lc_S_single,ls="--")
    # plt.errorbar(time_C,lc_C,yerr=sig_C,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    # plt.errorbar(time_S,lc_S,yerr=sig_S,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    # plt.errorbar(time_align,mag_align,yerr=err_align,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)

    # plt.scatter(time,lc,s=4,c="r",label = "lightcurve")
    plt.xlabel("time/HJD")
    plt.ylabel("magnitude")
    # plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/realfig/%04d.pdf"%(posi,))
    plt.close()


if __name__=="__main__":
    
    for i in range(1,50):
        try:
            testdraw2019(i)
        except:
            print(i,"error")
            print(traceback.format_exc())
            continue
