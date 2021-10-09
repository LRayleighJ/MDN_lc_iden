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
import MulensModel as mm

import netmodule.netGRUiden as lcnet

# initialize preload netparams
reload = 1
preload_Netmodel = "/home/zerui603/MDN_lc/GRUresnet_iden.pkl"

# initialize the storage path
rootval = "/scratch/zerui603/KMTiden_1d/val/"
samplepath = "/home/zerui603/MDN_lc/iden_1D/testnetfig/"


# initialize GPU
use_gpu = torch.cuda.is_available()
N_gpu = torch.cuda.device_count()
device_ids = range(N_gpu)
torch.backends.cudnn.benchmark = True

# initialize network
network = lcnet.ResNet()
if use_gpu:
    network = network.cuda()
    network = nn.DataParallel(network)
if reload == 1:
    network.load_state_dict(torch.load(preload_Netmodel))


# test single lc
# initialize the size of valsets
test_size = 2000
total_size = 60000

filelists = os.listdir(rootval)
random.shuffle(filelists)

for index_sample in range(test_size):
    datadir = list(np.load(rootval+filelists[index_sample], allow_pickle=True))
    labels = np.array(datadir[0],dtype=np.float64)

    lc_mag = np.array(datadir[3],dtype=np.float64)
    lc_mag = np.mean(np.sort(lc_mag)[-50:])-np.array(lc_mag)
    lc_mag_rnn = lc_mag.reshape((1000,1))
    lc_time_ori = np.array(datadir[1],dtype=np.float64)
    lc_time = (lc_time_ori-lc_time_ori[0])/(lc_time_ori[-1]-lc_time_ori[0])
    lc_time_rnn = lc_time.reshape((1000,1))
    lc_sig = np.array(datadir[4],dtype=np.float64)
    lc_sig_rnn = lc_sig.reshape((1000,1))*100

    data_input = np.concatenate((lc_mag_rnn,lc_time_rnn,lc_sig_rnn),axis=1)

    lg_q = np.log10(labels[2])
    lg_s = np.log10(labels[3])
    alpha = labels[4]
    u0 = labels[0]

    

    label = np.int(labels[-1])

    lc_data = np.array([data_input])

    print(lc_data.shape)
    print(label)


    inputs = torch.from_numpy(np.array([lc_data])).float()

    if use_gpu:
        inputs = inputs.cuda()
    network.eval()
    outputs = network(inputs).detach().cpu().numpy()[0]
    print(outputs)
    prediction = np.int(np.around(outputs[0]))
    plt.figure(figsize=(20,9))
    plt.subplot(121)
    plt.errorbar(lc_time_ori,lc_mag,yerr=lc_sig,fmt='o',capsize=2,elinewidth=1,ms=3,alpha=0.7,zorder=0)
    
    # plt.title("lg q=%.3f,lg s=%.3f,u0=%.3f,alpha=%.1f"%(lg_q,lg_s,labels[0],labels[4]))
    
    plt.subplot(122)
    ## trajectory
    ## Order of args: [u_0, rho, q, s, alpha, t_E]
    bl_model = mm.Model({'t_0': 0, 'u_0': labels[0],'t_E': labels[5], 'rho': labels[1], 'q': labels[2], 's': labels[3],'alpha': labels[4]})
    bl_model.set_default_magnification_method("VBBL")
    
    
    caustic = mm.Caustics(s=labels[3], q=labels[2])
    X_caustic,Y_caustic = caustic.get_caustics(n_points=2000)

    trace_x = -np.sin(labels[4]*np.pi/180)*labels[0]+lc_time_ori/labels[5]*np.cos(labels[4]*np.pi/180)
    trace_y = np.cos(labels[4]*np.pi/180)*labels[0]+lc_time_ori/labels[5]*np.sin(labels[4]*np.pi/180)
    
    plt.scatter(X_caustic,Y_caustic,s=1,c="b")
    plt.plot(trace_x,trace_y,c="g")
    plt.xlabel(r"$\theta_x$")
    plt.ylabel(r"$\theta_y$")
    plt.axis("scaled")
    plt.grid()
    plt.suptitle(r"$\log_{10} q$=%.3f,$\log_{10} s=$%.3f,$u_0=$%.3f,$\alpha=$%.1f,predicted=%d,label=%d"%(lg_q,lg_s,labels[0],labels[4],prediction,label),fontsize=30)
    plt.savefig(samplepath+str(index_sample)+"_lc.png")
    plt.close()

    # print("lightcurve %d has finished"%(index_sample,),datetime.datetime.now())
