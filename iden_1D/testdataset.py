import numpy as np
import matplotlib.pyplot as plt
import os
import random

num_test = 500
rootdir = "/scratch/zerui603/KMTiden_1d/val/"
storedir = "/home/zerui603/MDN_lc/iden_1D/testfig/"

rootdir2D = "/scratch/zerui603/KMTsimudata_iden/2Ddata/training/"
storedir2D = "/home/zerui603/MDN_lc/iden_1D/test2Dfig/"

filename_list = os.listdir(rootdir)
random.shuffle(filename_list)

filename_list2D = os.listdir(rootdir2D)
random.shuffle(filename_list2D)
'''
for i in range(num_test):
    data = list(np.load(rootdir+filename_list[i], allow_pickle=True))
    data_lc = list(np.array(data[1:],dtype=np.float64))
    labels = list(np.array(data[0],dtype=np.float64))

    # [u_0, rho, q, s, alpha, t_E, binary_or_not]
    # list(times),list(d_times),list(lc_noi_single),list(sig_single)

    time = data_lc[0]
    lc = data_lc[2]
    sig = data_lc[3]

    plt.figure()
    plt.errorbar(time,lc,yerr=sig,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    plt.xlabel("time/HJD")
    plt.ylabel("magnitude")
    plt.title("label:%d"%(labels[-1],))
    plt.gca().invert_yaxis()
    plt.savefig(storedir+str(i)+".png")
    plt.close()
'''

for i in range(num_test):
    data = list(np.load(rootdir2D+filename_list2D[i], allow_pickle=True))
    data_lc = list(np.array(data[1],dtype=np.float64))
    labels = list(np.array(data[0],dtype=np.float64))

    

    plt.figure()
    plt.imshow(data_lc, cmap='gray')
    plt.colorbar()
    plt.title("labels:%d"%(labels[-1],))
    plt.axis("scaled")
    plt.savefig(storedir2D+str(i)+".png")
    plt.close()




