import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import os

rootdir = "/scratch/zerui603/KMT_unet/high_ratio/training/"
name_list = os.listdir(rootdir)


size = 100
num_list = np.random.randint(0,1000,(size,))

for i in range(size):
    data = np.load(rootdir+name_list[num_list[i]],allow_pickle=True)
    ## [args_data, arg_singlefitting,time,d_time,lc_withnoi,err,lc_withoutnoi,lc_singlemodel,unet_label]

    lc_mag = np.array(data[4],dtype=np.float64)
    lc_bmodel = np.array(data[6],dtype=np.float64)
    lc_smodel= np.array(data[7],dtype=np.float64)
    
    lc_time = np.array(data[2],dtype=np.float64)

    lc_sig = np.array(data[5],dtype=np.float64)
    
    label = np.array(data[8]).astype(np.int64)

    plt.figure(figsize = (12,12))
    plt.subplot(211)
    plt.plot(lc_time,lc_smodel)
    plt.plot(lc_time,lc_bmodel)
    plt.errorbar(lc_time[label<0.5],lc_mag[label<0.5],yerr=lc_sig[label<0.5],fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5, color="b")
    plt.errorbar(lc_time[label>0.5],lc_mag[label>0.5],yerr=lc_sig[label>0.5],fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5, color="r")
    # plt.plot(time,fitdata.mag_cal(time,*args_fit))
    plt.xlabel("t/HJD-2450000",fontsize=16)
    plt.ylabel("Magnitude",fontsize=16)
    # plt.legend()
    plt.gca().invert_yaxis()

    plt.subplot(212)
    plt.scatter(lc_time,label,s=4)

    plt.savefig("/home/zerui603/MDN_lc_iden/unet/testfig/"+str(i)+".png")
    plt.close()

