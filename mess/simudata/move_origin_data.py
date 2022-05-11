import os
import multiprocessing as mp
import datetime
import numpy as np
import random
from scipy.optimize import curve_fit
import MulensModel as mm
import datamodule.data_hist as dhist
import datamodule.data_make2D as d2Ddata
import random

def kill_zero_0(x):
    if x>0:
        return 0
    else:
        return 1

kill_zero_np = np.frompyfunc(kill_zero_0,1,1)

def kill_zero(x):
    return kill_zero_np(x).astype(np.float64)

# path
rootpath_single = "/scratch/zerui603/KMTsimudata/sin_training_rename/"
rootpath_binary = "/scratch/zerui603/KMTsimudata/training/"
storedir_binary = "/scratch/zerui603/KMTsimudata_iden/training_binary/"
storedir_single = "/scratch/zerui603/KMTsimudata_iden/training_single/"
storedir_pure = "/scratch/zerui603/KMTsimudata_iden/training_single_pure/"
rootdir = rootpath_single
# arguments
size_ori_binary=2000000
num_single = 400000
num_binary = size_ori_binary-100

num_gen = 400000# num_binary

chi_limit = 10000

num_start=0

# count

def mag_cal(t,tE,t0,u0,fs,fb,m0):
    u = np.sqrt(((t-t0)/tE)**2+u0**2)
    A = (u**2+2)/(u*np.sqrt(u**2+4))
    return m0-2.5*np.log10(fs*A+fb)

def chi_square(single,binary,sigma):
    return np.sum(np.power((single-binary)/sigma,2))

def moveandcount(rootdir,storedir,num,num_start = 0):
    i=0
    single_count = 0
    binary_count = 0
    skipnum=0
    while True:
        try:
            data = list(np.load(rootdir+str(i+skipnum)+".npy", allow_pickle=True))
        except:
            print(i,"too little num of files")
            break
        data_lc = list(np.array(data[1:],dtype=np.float64))
        labels = list(np.array(data[0],dtype=np.float64))

        time = np.array(data_lc[0])
        lc = np.array(data_lc[2])
        lc_sig = np.array(data_lc[3])
        # [u_0, rho, q, s, alpha, t_E]
        u_0 = labels[0]
        rho = labels[1]
        q = labels[2]
        s = labels[3],
        alpha = labels[4]
        t_E = labels[5]
        basis_m = np.mean(np.sort(data_lc[2])[-50:])
        t_0 = 0

        left_bound = [0.5*t_E,t_0-t_E,u_0-1,1-0.5,0-0.5,0.8*basis_m]
        right_bound = [1.5*t_E,t_0+t_E,u_0+1,1+0.5,0+0.5,1.2*basis_m]


        try:
            popt, pcov = curve_fit(mag_cal,time,lc,bounds=(left_bound,right_bound))
        except:
            print(i,"optimizing error")
            skipnum += 1
            continue

        lc_fit = mag_cal(time,*popt)

        chi_s = chi_square(lc_fit,lc,lc_sig)

        if chi_s > chi_limit:
            labels.append(1)
            binary_count += 1
        else:
            labels.append(0)
            single_count += 1
        data_array=np.array([labels,list(data_lc)])
        np.save(storedir+str(i+num_start)+".npy",data_array,allow_pickle=True)
        i += 1
        if i > num:
            break 
    print("(total,binary,single):",i,binary_count,single_count)


def move_mp(index):
    single_count = 0
    binary_count = 0
    skipnum=0
    max_skipmum = 100
    while True:
        try:
            data = list(np.load(rootdir+str(index+skipnum)+".npy", allow_pickle=True))
        except:
            print(index,"file not found")
            break
        data_lc = list(np.array(data[1:],dtype=np.float64))
        labels = list(np.array(data[0],dtype=np.float64))

        time = np.array(data_lc[0])
        lc = np.array(data_lc[2])
        lc_sig = np.array(data_lc[3])
        # [u_0, rho, q, s, alpha, t_E]
        u_0 = labels[0]
        rho = labels[1]
        q = labels[2]
        s = labels[3],
        alpha = labels[4]
        t_E = labels[5]
        basis_m = np.mean(np.sort(data_lc[2])[-50:])
        t_0 = 0

        '''
        left_bound = [0.5*t_E,t_0-t_E,u_0-1,1-0.5,0-0.5,0.8*basis_m]
        right_bound = [1.5*t_E,t_0+t_E,u_0+1,1+0.5,0+0.5,1.2*basis_m]

        try:
            popt, pcov = curve_fit(mag_cal,time,lc,bounds=(left_bound,right_bound))
        except:
            print(index,"optimizing error")
            if skipnum > max_skipmum:
                break
            skipnum += 1
            continue
        
        lc_fit = mag_cal(time,*popt)

        chi_s = chi_square(lc_fit,lc,lc_sig)
        '''
        # labels.append(0)
        single_count += 1
        data_array=np.array([labels,list(data_lc)])
        np.save(storedir_pure+str(index+num_start)+".npy",data_array,allow_pickle=True)
        '''
        if chi_s > chi_limit:
            labels.append(1)
            binary_count += 1
            data_array=np.array([labels,list(data_lc)])
            np.save(storedir_binary+str(index+num_start)+".npy",data_array,allow_pickle=True)
        else:
            labels.append(0)
            single_count += 1
            data_array=np.array([labels,list(data_lc)])
            np.save(storedir_single+str(index+num_start)+".npy",data_array,allow_pickle=True)
        '''
        break

mixnum_bin=400000
mixnum_sin=400000
filename_list_bin = os.listdir(storedir_binary)
filename_list_sin = os.listdir(storedir_pure)
storedir_final = "/scratch/zerui603/KMTsimudata_iden/training_mixture/"
datastore = "/scratch/zerui603/KMTsimudata_iden/2Ddata/training/"
filename_list_final = os.listdir(storedir_final)
random.shuffle(filename_list_final)
filename_list_datastore = os.listdir(datastore)
random.shuffle(filename_list_datastore)


def mixture_mp(index):
    if index < mixnum_bin:
        command = "cp "+storedir_binary+filename_list_bin[index]+" "+storedir_final+str(int(index))+".npy"
        os.system(command)
    else:
        command = "cp "+storedir_pure+filename_list_sin[index-mixnum_bin]+" "+storedir_final+str(int(index))+".npy"
        os.system(command)


def make2Ddata(index):
    data = list(np.load(storedir_final+filename_list_final[index], allow_pickle=True))
    data_lc = list(np.array(data[1],dtype=np.float64))
    labels = list(np.array(data[0],dtype=np.float64))

    time = np.array(data_lc[0])
    lc = np.array(data_lc[2])
    lc_sig = np.array(data_lc[3])
    lc_sig = np.mean(lc_sig)*kill_zero(lc_sig)+lc_sig
    # [u_0, rho, q, s, alpha, t_E]
    u_0 = labels[0]
    rho = labels[1]
    q = labels[2]
    s = labels[3]
    alpha = labels[4]
    t_E = labels[5]
    basis_m = np.mean(np.sort(data_lc[2])[-50:])
    t_0 = 0

    """
    data_lc = list(np.array(data[1:],dtype=np.float64))
    labels = list(np.array(data[0],dtype=np.float64))

    lc_withnoi = np.mean(np.sort(data_lc[2])[-50:])-np.array(data_lc[2])
    lc_sig = np.array(data_lc[3])    
    lc_time = np.array(data_lc[0])
    """

    left_bound = [0.5*t_E,t_0-t_E,u_0-1,1-0.5,0-0.5,0.8*basis_m]
    right_bound = [1.5*t_E,t_0+t_E,u_0+1,1+0.5,0+0.5,1.2*basis_m]
    

    # use lc,time,lc_sig

    mag_ratio = 2

    mag_sort = np.sort(lc)

    mag_leftlim = np.mean(mag_sort[:50])-np.std(mag_sort[:50])*mag_ratio
    mag_rightlim = np.mean(mag_sort[-50:])+np.std(mag_sort[-50:])*0.5
    
    
    figdata = d2Ddata.get2Ddensity(time=time,mag=lc,sigma=lc_sig,mag_leftlim=mag_leftlim,mag_rightlim=mag_rightlim,dim=150)
    # nodatapixel:0
    figdata = 1-np.tanh(3*figdata)

    data_array=np.array([labels,list(figdata)])
    np.save(datastore+str(index)+".npy",data_array,allow_pickle=True)

    
    del data_array
    return 0

def check_data(index):
    data = list(np.load(storedir_final+str(index)+".npy", allow_pickle=True))
    data_lc = list(np.array(data[1],dtype=np.float64))
    labels = list(np.array(data[0],dtype=np.float64))
    if labels[-1] != 0 and labels[-1] != 1:
        print(index,labels)
    if np.isnan(np.sum(labels)):
        print("NAN!label", index)
    
    if np.isnan(np.sum(data_lc)):
        print("NAN!datalc", index)
    

if __name__=="__main__":
    u = 0
    starttime = datetime.datetime.now()
    print("starttime:",starttime)
    print("cpu_count:",os.cpu_count())
    with mp.Pool(20) as p:
        p.map(make2Ddata, range(800000))
    endtime = datetime.datetime.now()
    print("end time:",endtime)
    print("total:",endtime - starttime)
    starttime = datetime.datetime.now()
    print("starttime:",starttime)
    print("cpu_count:",os.cpu_count())
    with mp.Pool(20) as p:
        p.map(check_data, range(800000))
    endtime = datetime.datetime.now()
    print("end time:",endtime)
    print("total:",endtime - starttime)
