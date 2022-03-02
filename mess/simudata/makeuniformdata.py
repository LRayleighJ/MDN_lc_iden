import os
import multiprocessing as mp
import datetime
import numpy as np
import random
import matplotlib.pyplot as plt


datadir = "/scratch/zerui603/KMTsimudata/test_smooth/"
targetdir = "/scratch/zerui603/KMTsimudata/test_smooth/"

size_total = 1743

num_bin = 12

cadence = 4/num_bin

posi_stat = np.zeros(num_bin)

num = 0

lg_q_data = []

for index in range(size_total):
    data = list(np.load(datadir+str(index)+".npy", allow_pickle=True))
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

    # Order of args: [u_0, rho, q, s, alpha, t_E]

    lg_q = np.log10(labels[2])
    print(lg_q)
    lg_q_data.append(lg_q)
    

plt.figure()
plt.hist(lg_q_data, bins=np.linspace(-4,0,13), facecolor=(0, 127/255, 194/255, 1), edgecolor=(0, 127/255, 194/255, 0.35))
plt.xlabel(r"$\log_{10} q$")
plt.ylabel(r"Number of Event")
plt.title(r"$\log_{10}q$ Distribution of Test Dataset")
plt.savefig("qdistribution.png")


