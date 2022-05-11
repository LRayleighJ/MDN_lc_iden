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

import netmodule.netMDN_ResNet2D_iden as lcnet

# initialize preload netparams
reload = 1
preload_Netmodel = "/home/zerui603/MDN_lc/resnet_params2D4args_iden.pkl"

# initialize the storage path
rootval = "/scratch/zerui603/KMTsimudata_iden/2Ddata/training/"
samplepath = "/home/zerui603/MDN_lc/iden_2D/test2Dnetfig/"


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

    lc_mag = np.array(datadir[1],dtype=np.float64)

    lg_q = np.log10(labels[2])
    lg_s = np.log10(labels[3])
    alpha = labels[4]
    u0 = labels[0]

    # q_label = lg_q/4
    # s_label = (lg_s-np.log10(0.3))/(np.log10(3)-np.log10(0.3))
    # alpha_label = alpha/360
    # u0_label = u0
    # ux_label = (u0*np.cos(np.pi/180*alpha)+1)/2
    # uy_label = (u0*np.sin(np.pi/180*alpha)+1)/2
    

    label = np.int(labels[-1])


    inputs = torch.from_numpy(np.array([[lc_mag]])).float()

    if use_gpu:
        inputs = inputs.cuda()
    network.eval()
    outputs = network(inputs).detach().cpu().numpy()[0]
    print(outputs)
    prediction = np.int(np.around(outputs[0]))


    plt.figure()
    plt.imshow(lc_mag, cmap='gray')
    plt.colorbar()
    plt.suptitle(r"$\log_{10} q$=%.3f,$\log_{10} s=$%.3f,$u_0=$%.3f,$\alpha=$%.1f,predicted=%d,label=%d"%(lg_q,lg_s,labels[0],labels[4],prediction,label))
    plt.savefig(samplepath+str(index_sample)+"_lc.png")
    plt.close()

    # print("lightcurve %d has finished"%(index_sample,),datetime.datetime.now())
