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

rootdir = "/scratch/zerui603/KMT_simu_lowratio/qseries/10to15testbins/0_5/"
targetdir = "/home/zerui603/MDN_lc/datagenerate/testfig/"

def fit_npy(index):
    path_temp = rootdir+str(index)+".npy"
    data = list(np.load(path_temp,allow_pickle=True))
    
    labels = list(np.array(data[0],dtype=np.float64))
    print(labels)
    ## [u_0, rho, q, s, alpha, t_E, basis_m, t_0, dchi^2, label]
    time = data[1]
    lc = np.array(data[3])
    sig = np.array(data[4])
    lc_single = np.array(data[7])
    lc_model = np.array(data[5])
    chi_curve = np.array(data[8])

    mag_max_lim = np.mean(np.sort(lc_model)[-25:])
    mag_min_lim = np.mean(np.sort(lc_model)[:25])
    mag_max_lim += 0.1*(mag_max_lim-mag_min_lim)
    mag_min_lim -= 0.1*(mag_max_lim-mag_min_lim)

    plt.figure(figsize=[12,21])
    plt.subplot(311)
    plt.errorbar(time,lc,yerr=sig,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
    plt.scatter(time,lc,s=4,c="r",label = "lightcurve")
    plt.plot(time,lc_single,label="single fit")
    plt.plot(time,lc_model,label="model")
    plt.xlabel("time/HJD")
    plt.ylabel("magnitude")
    plt.ylim(mag_min_lim,mag_max_lim)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.subplot(312)
    plt.scatter(time,chi_curve**2,s=2)
    plt.plot(time,0*np.array(time),linestyle="--",c="g")
    plt.plot(time,1+0*np.array(time),linestyle="--",c="r")
    plt.xlabel("time/HJD")
    plt.ylabel(r"$(\frac{m_i-m_{single}}{\sigma_i})^2$")
    
    plt.subplot(313)
    plt.scatter(time,chi_curve**2-((lc-lc_model)/sig)**2,s=2)
    plt.plot(time,0*np.array(time),linestyle="--",c="g")
    plt.plot(time,1+0*np.array(time),linestyle="--",c="r")
    plt.xlabel("time/HJD")
    plt.ylabel(r"$(\frac{m_i-m_{single}}{\sigma_i})^2-(\frac{m_i-m_{model}}{\sigma_i})^2$")


    if labels[-1] == 1:
        plt.suptitle(r"$\log_{10} q$=%.3f,$\log_{10} s=$%.3f,$u_0=$%.3f,$\alpha=$%.1f,$\Delta \chi^2$:%.3f,index=%d"%(np.log10(labels[2]),np.log10(labels[3]),labels[0],labels[4],labels[-2],index,))
    else:
        plt.suptitle("$\Delta \chi^2$:%.3f,index=%d"%(labels[-1],index,))


    plt.savefig(targetdir+str(index)+"_lc.png")
    plt.close()

    

    print(str(index),labels[-3],labels[-2])
    return 

if __name__=="__main__":
    starttime = datetime.datetime.now()
    with mp.Pool(20) as p:
        p.map(fit_npy, range(300))
    endtime = datetime.datetime.now()