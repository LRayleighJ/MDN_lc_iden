import numpy as np
import matplotlib.pyplot as plt
import datamodule.data_hist as dhist
import datamodule.data_make2D as d2Ddata
import datetime
import multiprocessing as mp
import os
from scipy.optimize import curve_fit
import MulensModel as mm

dataroot = "/scratch/zerui603/KMTsimudata/sin_training_rename/"

datastore = "/scratch/zerui603/KMTsimu2D_identify/training/"

def mag_cal(t,tE,t0,u0,fs,fb,m0):
    u = np.sqrt(((t-t0)/tE)**2+u0**2)
    A = (u**2+2)/(u*np.sqrt(u**2+4))
    return m0-2.5*np.log10(fs*A+fb)

def chi_square(single,binary,sigma):
    return np.sum(np.power((single-binary)/sigma,2))

def make2Ddata(index):
    data = list(np.load(dataroot+str(index)+".npy", allow_pickle=True))
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

    """
    data_lc = list(np.array(data[1:],dtype=np.float64))
    labels = list(np.array(data[0],dtype=np.float64))

    lc_withnoi = np.mean(np.sort(data_lc[2])[-50:])-np.array(data_lc[2])
    lc_sig = np.array(data_lc[3])    
    lc_time = np.array(data_lc[0])
    """

    left_bound = [0.5*t_E,t_0-t_E,u_0-1,1-0.5,0-0.5,0.8*basis_m]
    right_bound = [1.5*t_E,t_0+t_E,u_0+1,1+0.5,0+0.5,1.2*basis_m]
    '''
    # fit 
    try:
        popt, pcov = curve_fit(mag_cal,time,lc,bounds=(left_bound,right_bound))
    except:
        print(index,"Error")
        return 0
    lc_fit = mag_cal(time,*popt)

    chi_s = chi_square(lc_fit,lc,lc_sig)

    if chi_s > 10000:
        labels.append(1)
    else:
    '''
    labels.append(0)

    # use lc,time,lc_sig

    mag_ratio = 2

    mag_sort = np.sort(lc)

    mag_leftlim = np.mean(mag_sort[:50])-np.std(mag_sort[:50])*mag_ratio
    mag_rightlim = np.mean(mag_sort[-50:])+np.std(mag_sort[-50:])*0.5
    
    
    figdata = d2Ddata.get2Dmatrix(time=time,mag=lc,sigma=lc_sig,mag_leftlim=mag_leftlim,mag_rightlim=mag_rightlim,dim=150)
    # nodatapixel:0
    figdata = (1-figdata)

    data_array=np.array([labels,list(figdata)])
    np.save(datastore+str(index)+".npy",data_array,allow_pickle=True)

    print("lc %d has finished"%(index))
    print(datetime.datetime.now())
    del data_array
    return 0


if __name__=="__main__":
    u = 0
    num_batch = 400000
    starttime = datetime.datetime.now()
    print("starttime:",starttime)
    print("cpu_count:",os.cpu_count())
    with mp.Pool() as p:
        p.map(make2Ddata, range(u, u+num_batch))
    endtime = datetime.datetime.now()
    print("end time:",endtime)
    print("total:",endtime - starttime)


