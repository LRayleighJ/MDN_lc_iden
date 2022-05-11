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
import datetime
import MulensModel as mm
import scipy.signal as signal
import scipy.fft
# from torchviz import make_dot 

import netmodule.netMDN_Resnet2D as lcnet
import datamodule.datahist as dhist
import datamodule.data_make2D as d2Ddata

# initialize preload netparams
reload = 1
preload_Netmodel = "/home/zerui603/MDN_lc/testnet/netparams/resnet_params2D4args.pkl"

# initialize the storage path
rootval = "/scratch/zerui603/KMTsimudata/test/"
samplepath = "/home/zerui603/MDN_lc/testnet/testfig/"


# initialize GPU
use_gpu = torch.cuda.is_available()
N_gpu = torch.cuda.device_count()
device_ids = range(N_gpu)
torch.backends.cudnn.benchmark = True

# initialize network
network = lcnet.ResNet()
criterion = lcnet.Loss_fn()
if use_gpu:
    network = network.cuda()
    criterion = criterion.cuda()
    network = nn.DataParallel(network)
if reload == 1:
    network.load_state_dict(torch.load(preload_Netmodel))


# test single lc
# initialize the size of valsets
test_size = 500
total_size = 16000
n_oripoints = 1000

resample_size = 100
remodel_size = 5

index_list = np.random.randint(total_size,size=test_size)

# initialize the range of args-space
## Attention: the args that input into network are multiplied by 5

range_lgq = 1
range_lgs = 0.3
range_ux = 0.5
range_uy = 0.5

for index_sample in range(test_size):
    data = list(np.load(rootval+str(index_list[index_sample])+".npy", allow_pickle=True))
    data_lc = list(np.array(data[1:],dtype=np.float64))
    labels = list(np.array(data[0],dtype=np.float64))
    # /KMTsimudata: time,dtime,mag,noise,errorbar
    
    lc_base = np.mean(np.sort(data_lc[2])[-50:])

    lc = lc_base-np.array(data_lc[2])
    lc_sig = np.array(data_lc[4])    
    lc_time = np.array(data_lc[0])
    # lc_dtime = np.array(data_lc[1])
    lc_noi = np.array(data_lc[3])
    lc_withnoi = lc  +lc_noi

    # raise RuntimeError("Test")

    # Order of args: [u_0, rho, q, s, alpha, t_E]
    lg_q = np.log10(labels[2])
    lg_s = np.log10(labels[3])
    ux = labels[0]*np.cos(np.pi/180*labels[4])
    uy = labels[0]*np.sin(np.pi/180*labels[4])
    realargs = [lg_q,lg_s,ux,uy]

    # origin histed lightcurve
    mag_ratio = 2

    mag_sort = np.sort(lc_withnoi)

    mag_leftlim = np.mean(mag_sort[:50])-np.std(mag_sort[:50])*mag_ratio
    mag_rightlim = np.mean(mag_sort[-50:])+np.std(mag_sort[-50:])*0.5
    
    
    figdata = d2Ddata.get2Dmatrix(time=lc_time,mag=lc_withnoi,sigma=lc_sig,mag_leftlim=mag_leftlim,mag_rightlim=mag_rightlim,dim=150)
    # nodatapixel:0
    figdata = (1-figdata)


    inputs = torch.from_numpy(np.array([[figdata]])).float()

    if use_gpu:
        inputs = inputs.cuda()
    network.eval()
    # Order of outputs: q,s,ux,uy
    # Silly code......
    pi1,pi2,pi3,pi4,mu1,mu2,mu3,mu4,sigma1,sigma2,sigma3,sigma4 = network(inputs)
    pi1 = pi1.detach().cpu().numpy()[0]
    pi2 = pi2.detach().cpu().numpy()[0]
    pi3 = pi3.detach().cpu().numpy()[0]
    pi4 = pi4.detach().cpu().numpy()[0]
    pis = np.exp(np.array([pi1,pi2,pi3,pi4]))
    mu1 = mu1.detach().cpu().numpy()[0]
    mu2 = mu2.detach().cpu().numpy()[0]
    mu3 = mu3.detach().cpu().numpy()[0]
    mu4 = mu4.detach().cpu().numpy()[0]
    mus = np.array([mu1,mu2,mu3,mu4])
    sigma1 = sigma1.detach().cpu().numpy()[0]
    sigma2 = sigma2.detach().cpu().numpy()[0]
    sigma3 = sigma3.detach().cpu().numpy()[0]
    sigma4 = sigma4.detach().cpu().numpy()[0]
    sigmas = np.array([sigma1,sigma2,sigma3,sigma4])
    # calc the range of args
    lgq_domain = np.linspace(lg_q-range_lgq,lg_q+range_lgq,200)
    lgs_domain = np.linspace(-1*np.abs(lg_s)-range_lgs,np.abs(lg_s)+range_lgs,200)
    ux_domain = np.linspace(ux-range_ux,ux+range_ux,200)
    uy_domain = np.linspace(uy-range_uy,uy+range_uy,200)

    args_domains = np.array([lgq_domain,lgs_domain,ux_domain,uy_domain])
    axislabels = ["lg q","lg s","ux","uy"]

    # Plot density figure
    mat_id = (np.array(range(3*3))+1).reshape((3,3)).T
    plot_base = 330

    plt.figure(figsize=(24,24)) 
    for index_i in range(3):
        for index_j in range(index_i+1,4):
            x_domain = dhist.tran_universal(args_domains[index_i],index_i)
            y_domain = dhist.tran_universal(args_domains[index_j],index_j)
            X,Y = np.meshgrid(x_domain,y_domain)
            x_prob = dhist.multi_gaussian_prob(pis[index_i],mus[index_i],sigmas[index_i])(X)
            y_prob = dhist.multi_gaussian_prob(pis[index_j],mus[index_j],sigmas[index_j])(Y)
            xy_prob = x_prob*y_prob
            plt.subplot(plot_base+mat_id[index_i][index_j-1])
            plt.contourf(args_domains[index_i], args_domains[index_j], xy_prob)
            plt.scatter([realargs[index_i]],[realargs[index_j]],s=120,c="r",marker="x")
            plt.xlabel(axislabels[index_i],fontsize=30)
            plt.ylabel(axislabels[index_j],fontsize=30)
            # plt.axis("equal")
            plt.grid()
    plt.suptitle(r"$\lg q$=%.3f,$\lg s=$%.3f,$u_0=$%.3f,$\alpha=$%.1f,ID=%d"%(lg_q,lg_s,labels[0],labels[4],index_list[index_sample]),fontsize=40)
    plt.savefig(samplepath+str(index_sample)+".png")
    plt.close()
    
    '''
    # Lomb-scargle
    n_point_lomb=10000
    freq_range = np.linspace(0.5,20,n_point_lomb)
    freq_spectrum = signal.lombscargle(lc_time, lc_withnoi, freq_range, normalize=False)
    freq_spectrum = np.abs(freq_spectrum)**2
    ifft_lc = scipy.fft.ifft(freq_spectrum)
    time_scaled_ls = np.linspace(lc_time[0],lc_time[-1],len(ifft_lc))

    plt.figure(figsize=(10,6))
    plt.plot(freq_range,freq_spectrum)
    plt.xlabel("freq")
    plt.savefig(samplepath+str(index_sample)+"_freq_spectrum.png")
    plt.close()
    '''

    plt.figure(figsize=(20,9))
    plt.subplot(121)
    plt.errorbar(lc_time,lc_withnoi,yerr=lc_sig,fmt='o:',capsize=2,elinewidth=1,ms=1,zorder=0)
    plt.title("lg q=%.3f,lg s=%.3f,u0=%.3f,alpha=%.1f,ID=%d"%(lg_q,lg_s,labels[0],labels[4],index_list[index_sample]))
    
    plt.subplot(122)
    ## trajectory
    ## Order of args: [u_0, rho, q, s, alpha, t_E]
    bl_model = mm.Model({'t_0': 0, 'u_0': labels[0],'t_E': labels[5], 'rho': labels[1], 'q': labels[2], 's': labels[3],'alpha': labels[4]})
    bl_model.set_default_magnification_method("VBBL")
    
    
    caustic = mm.Caustics(s=labels[3], q=labels[2])
    X_caustic,Y_caustic = caustic.get_caustics(n_points=2000)

    trace_x = -np.sin(labels[4]*np.pi/180)*labels[0]+lc_time/labels[5]*np.cos(labels[4]*np.pi/180)
    trace_y = np.cos(labels[4]*np.pi/180)*labels[0]+lc_time/labels[5]*np.sin(labels[4]*np.pi/180)
    
    plt.scatter(X_caustic,Y_caustic,s=1,c="b")
    plt.plot(trace_x,trace_y,c="g")
    plt.xlabel(r"$\theta_x$")
    plt.ylabel(r"$\theta_y$")
    plt.axis("scaled")
    plt.grid()
    plt.suptitle(r"$\lg q$=%.3f,$\lg s=$%.3f,$u_0=$%.3f,$\alpha=$%.1f,ID=%d"%(lg_q,lg_s,labels[0],labels[4],index_list[index_sample]),fontsize=30)
    plt.savefig(samplepath+str(index_sample)+"_lc.png")
    plt.close()

    print("lightcurve %d has finished"%(index_sample,),datetime.datetime.now())
    

