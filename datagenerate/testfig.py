import os
import multiprocessing as mp
import datetime
import numpy as np
import random
import scipy.optimize as op
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def mag_cal(t,tE,t0,u0,m0):
    u = np.sqrt(((t-t0)/tE)**2+u0**2)
    A = (u**2+2)/(u*np.sqrt(u**2+4))
    return m0-2.5*np.log10(A)

def chi_square(single,binary,sigma):
    return np.sum(np.power((single-binary)/sigma,2))

def chi2_for_minimize(args,time,mag,sigma):
    return chi_square(mag_cal(time,*args),mag,sigma)


def chi2_for_model(theta, event, parameters_to_fit):
    """for given event set attributes from parameters_to_fit
    (list of str) to values from the theta list"""
    for (key, parameter) in enumerate(parameters_to_fit):
        setattr(event.model.parameters, parameter, theta[key])
    return event.get_chi2()

rootdir = "/scratch/zerui603/KMT_simu_lowratio/qseries/10to15longtest/"
targetdir = "/home/zerui603/MDN_lc/datagenerate/testfig/"

def fit_npy(index):
    path_temp = rootdir+str(index)+".npy"
    data = list(np.load(path_temp,allow_pickle=True))
    
    labels = list(np.array(data[0],dtype=np.float64))
    ## [u_0, rho, q, s, alpha, t_E, basis_m, t_0, chi^2,chi^2_test, label]
    ## args_data_test,list(times),list(d_times),list(lc_noi),list(sig),list(magnitude_tran(lc,basis_m)),list(args_minimize),list(args_minimize_fortest),list(lc_fit_minimize),list(lc_fit_minimize_fortest)
    time = data[1]
    lc = np.array(data[3])
    sig = np.array(data[4])
    lc_single = np.array(data[8])
    lc_model = np.array(data[5])

    lc_single_lesspointfit = np.array(data[9])


    mag_max_lim = np.mean(np.sort(lc_model)[-25:])
    mag_min_lim = np.mean(np.sort(lc_model)[:25])
    mag_max_lim += 0.1*(mag_max_lim-mag_min_lim)
    mag_min_lim -= 0.1*(mag_max_lim-mag_min_lim)

    plt.figure(figsize=[12,8])
    
    plt.errorbar(time,lc,yerr=sig,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    plt.scatter(time,lc,s=4,c="r",label = "lightcurve")
    plt.plot(time,lc_single,label="single fit")
    plt.plot(time,lc_single_lesspointfit,label="single fit (less points for fitting)")
    plt.plot(time,lc_model,label="model")
    plt.xlabel("time/HJD")
    plt.ylabel("magnitude")
    plt.ylim(mag_min_lim,mag_max_lim)
    plt.legend()
    plt.gca().invert_yaxis()
    

    if labels[-1] == 1:
        plt.suptitle(r"$\log_{10} q$=%.3f,$\log_{10} s=$%.3f,$u_0=$%.3f,$\alpha=$%.1f,$\Delta \chi^2$:%.3f,$\Delta \chi^2$(less fitting):%.3f"%(np.log10(labels[2]),np.log10(labels[3]),labels[0],labels[4],labels[-3],labels[-2],))
    else:
        plt.suptitle("$\Delta \chi^2$:%.3f,$\Delta \chi^2$(less fitting):%.3f"%(labels[-3],labels[-2],))

    plt.savefig(targetdir+str(index)+"_lc.png")
    plt.close()

    

    print(str(index),labels[-3],labels[-2])
    return 

if __name__=="__main__":
    starttime = datetime.datetime.now()
    with mp.Pool(20) as p:
        p.map(fit_npy, range(300))
    endtime = datetime.datetime.now()