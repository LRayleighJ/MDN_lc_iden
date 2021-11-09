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

rootdir = "/scratch/zerui603/KMT_simu_lowratio/training/"
targetdir = "/home/zerui603/MDN_lc/datagenerate/testfig/"

def fit_npy(index):
    path_temp = rootdir+str(index)+".npy"
    data = np.load(path_temp,allow_pickle=True)
    data_lc = np.array(data[1:])
    label = data[0]
    # [u_0, rho, q, s, alpha, t_E, basis_m, t_0,0]
    #print([len(x) for x in data_lc])
    
    time = np.array(data_lc[0])
    mag = np.array(data_lc[2])
    errorbar = np.array(data_lc[3])
    mag_clean = np.array(data_lc[4])
    #try:
    initial_guess = [label[5],label[7],label[0],label[6]]


    result = op.minimize(chi2_for_minimize, x0=initial_guess,args=(time,mag,errorbar), method='Nelder-Mead')
    # print("Fitting was successful? {:}".format(result.success))
    if not result.success:
        print(result.message,index)
        return 
    # print("Function evaluations: {:}".format(result.nfev))
    if isinstance(result.fun, np.ndarray):
        if result.fun.ndim == 0:
            result_fun = float(result.fun)
        else:
            result_fun = result.fun[0]
    else:
        result_fun = result.fun

    args_minimize = result.x.tolist()
    lc_fit_minimize = mag_cal(time,*args_minimize)

    chi_s_minimize = chi_square(lc_fit_minimize,mag,errorbar)
    chi_s_binary = chi_square(mag_clean,mag,errorbar)
    print(index,"chi^2 for minimize: ",chi_s_minimize)

    plt.figure(figsize=(20,12))
    plt.errorbar(time,mag,yerr=errorbar,fmt='o',capsize=2,elinewidth=1,ms=3,alpha=0.7,zorder=0)
    plt.plot(time,mag_clean,label="mag_clean")
    plt.plot(time,lc_fit_minimize,label="single fit")
    plt.gca().invert_yaxis()
    plt.title("[u_0=%.3f, rho=%.4f, q=%.7f, s=%.3f, alpha=%.2f, t_E=%.2f, basis_m=%.2f, t_0=%.2f,label=%d,$\Delta\chi^2$=%.3f"%(label[0],label[1],label[2],label[3],label[4],label[5],label[6],label[7],label[8],chi_s_minimize-chi_s_binary))
    plt.savefig(targetdir+str(index)+"_lc.png")
    plt.close()

    '''
    except:
        print("Minimize error",index)
        return 
    '''
    return 

if __name__=="__main__":
    starttime = datetime.datetime.now()
    with mp.Pool(20) as p:
        p.map(fit_npy, range(300))
    endtime = datetime.datetime.now()