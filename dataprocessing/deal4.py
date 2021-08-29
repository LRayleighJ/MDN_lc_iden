import numpy as np
import matplotlib.pyplot as plt
import datamodule.data_hist as dhist
import datetime
import multiprocessing as mp
import os

rootdir = "/scratch/zerui603/KMTsimudata/training/"
storedir = "/scratch/zerui603/KMTsimudata200/training/"

# rootval = "/scratch/zerui603/KMTsimudata/val/"
# samplepath = "/home/zerui603/iden_lc_ML_ver2/data_fig/"

# index_list = np.random.randint(total_size,size=test_size)

def lightcurve_hist(index_sample):
    data = list(np.load(rootdir+str(index_sample)+".npy", allow_pickle=True))
    data_lc = list(np.array(data[1:],dtype=np.float64))
    labels = list(np.array(data[0],dtype=np.float64))

    lc_withnoi = np.mean(np.sort(data_lc[2])[-50:])-np.array(data_lc[2])
    lc_sig = np.array(data_lc[3])    
    lc_time = np.array(data_lc[0])
    # lc_dtime = np.array(data_lc[1])

    labels.append(lc_time[0])
    labels.append(lc_time[-1])

    # time_ref = np.linspace(lc_time[0],lc_time[-1],200)

    lc_ref = dhist.lc_hist(lc_withnoi,lc_time,lc_sig)

    data_array=np.array([labels,list(lc_ref)])
    np.save(storedir+str(index_sample)+".npy",data_array,allow_pickle=True)

    print("lc %d has finished"%(index_sample))
    print(datetime.datetime.now())

if __name__=="__main__":
    u = 3*500000
    num_batch = 500000
    starttime = datetime.datetime.now()
    print("starttime:",starttime)
    print("cpu_count:",os.cpu_count())
    with mp.Pool() as p:
        p.map(lightcurve_hist, range(u, u+num_batch))
    endtime = datetime.datetime.now()
    print("end time:",endtime)
    print("total:",endtime - starttime)



