import MulensModel as mm
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import re
import pandas as pd
import scipy.optimize as op
from scipy.optimize import curve_fit

info_2019 = np.loadtxt("/home/zerui603/MDN_lc_iden/unet/datamodule/list_2019.dat",dtype=str,usecols=range(14))

def get_test_time_series(t):
    return np.linspace(np.min(t),np.max(t),100)

def KMTmagDflux(dflux,Icat):
    return -5/2*np.log10(-6.309491884030959e-12*(dflux)+10**(-2/5*Icat))

def KMTflux(mag):
    return 10**(-2/5*mag)/(6.309491884030959e-12)

def KMTmag(flux):
    return -5/2*np.log10(6.309491884030959e-12*(flux))

def chi_square(single,binary,sigma):
    return np.sum(np.power((single-binary)/sigma,2))


def killnan(data):
    return data[:, ~np.isnan(data).any(axis=0)]

def killinf(data):
    return data[:, ~np.isinf(data).any(axis=0)]

def killstr(data):
    return data[~(data == "___").any(axis=1)].astype(np.float)



def cut(t_0,t_e,data):
    new_data = []
    for array_sample in data:
        if array_sample[0]<t_0+2*t_e and array_sample[0]>t_0-2*t_e:
            if array_sample[-3]<8:
                new_data.append(array_sample)
    return np.array(new_data)

def killbinweirdpoint(lc,length = 1000, bins = 100,ratio = 2):
    size_bin = np.int(length//bins)
    new_lc = []
    new_posi = []
    drop_posi = []
    for i in range(bins):
        lc_piece = lc[np.int(size_bin*i): np.int(size_bin*(i+1))]
        mean_piece = np.mean(lc_piece)
        std_piece = np.std(lc_piece)
        
        for j in range(len(lc_piece)):
            if np.abs(lc_piece[j]-mean_piece) < ratio*std_piece:
                new_lc.append(lc_piece[j])
                new_posi.append(np.int(size_bin*i+j))
            else:
                drop_posi.append(np.int(size_bin*i+j))
    return np.array(new_lc),np.array(new_posi),np.array(drop_posi)

# getfile

def getKMTIfilelist(rootdir = "/scratch/zerui603/KMTrealdata/",year=2018,posi=1):
    path = rootdir+"%d_%04d/"%(year,posi,)
    filelist_origin = os.listdir(path)

    filelist = [filename for filename in filelist_origin if re.search("KMT.*_I.pysis",filename)]
    filelist_A = [filename for filename in filelist if filename[3]=="A"]
    filelist_C = [filename for filename in filelist if filename[3]=="C"]
    filelist_S = [filename for filename in filelist if filename[3]=="S"]

    return filelist_A,filelist_C,filelist_S

def getKMTdata(rootdir = "/scratch/zerui603/KMTrealdata/",year=2019,posi=1,cutratio=[-2,2],err_threshold=0.5,FWHM_threshold=10,sky_threshold=10000,inverse=False):
    ## [HJD  \Delta_flux flux_err  mag  mag_err  fwhm  sky  secz]
    # data_args = pd.DataFrame(data=np.load("KMT_args.npy",allow_pickle=True))
    # KMT_official_args = data_args.loc[data_args["index"]=="%d_%04d"%(year,posi,)].values[0]

    info_event = info_2019[posi-1]
    t0_model = np.float(info_event[6])
    tE_model = np.float(info_event[7])
    u0_model = np.float(info_event[8])
    Is_model = np.float(info_event[9])
    Ib_model = np.float(info_event[10])
    m0_model = np.float(info_event[11])

    Fs = KMTflux(Is_model)
    Fb = KMTflux(Ib_model)


    # tE, t0, u0, m0, fb
    KMT_official_args = np.array([tE_model,t0_model,u0_model,m0_model,1-Fs/Fb])

    t_0_KMT = t0_model
    t_E_KMT = tE_model
    u_0_KMT = u0_model
    Ibase_KMT = Ib_model

    path = rootdir+"%d_%04d/"%(year,posi,)
    filelistI_A,filelistI_C,filelistI_S = getKMTIfilelist(rootdir,year,posi)
    datas_A = [killstr(np.loadtxt(path+filename,dtype=str)).T for filename in filelistI_A]
    datas_C = [killstr(np.loadtxt(path+filename,dtype=str)).T for filename in filelistI_C]
    datas_S = [killstr(np.loadtxt(path+filename,dtype=str)).T for filename in filelistI_S]
    data_A = np.hstack(datas_A)
    data_C = np.hstack(datas_C)
    data_S = np.hstack(datas_S)


    # print(data_A.shape)

    data = np.c_[data_A,data_C,data_S]
    data_index = np.argwhere((data[4]<err_threshold)&(data[5]<FWHM_threshold)&(data[5]>0)&(data[6]<sky_threshold)).T[0]
    time = data[0][data_index]
    mag = data[3][data_index]
    errorbar = data[4][data_index]

    order = np.argsort(time)
    time = time[order]
    mag = mag[order]
    errorbar = errorbar[order]

    if inverse == False:
        time_select_index = np.argwhere((time<t_0_KMT+cutratio[1]*t_E_KMT)&(time>t_0_KMT+cutratio[0]*t_E_KMT)).T[0]
        bound_cut = [time_select_index[0],time_select_index[-1]]
        time = time[time_select_index]
        mag = mag[time_select_index]
        errorbar = errorbar[time_select_index]
        # print(len(time))
        data_A_index = np.argwhere((data_A[4]<err_threshold)&(data_A[0]<t_0_KMT+cutratio[1]*t_E_KMT)&(data_A[0]>t_0_KMT+cutratio[0]*t_E_KMT)&(data_A[5]<FWHM_threshold)&(data_A[5]>0)&(data_A[6]<sky_threshold)).T[0]
        data_C_index = np.argwhere((data_C[4]<err_threshold)&(data_C[0]<t_0_KMT+cutratio[1]*t_E_KMT)&(data_C[0]>t_0_KMT+cutratio[0]*t_E_KMT)&(data_C[5]<FWHM_threshold)&(data_C[5]>0)&(data_C[6]<sky_threshold)).T[0]
        data_S_index = np.argwhere((data_S[4]<err_threshold)&(data_S[0]<t_0_KMT+cutratio[1]*t_E_KMT)&(data_S[0]>t_0_KMT+cutratio[0]*t_E_KMT)&(data_S[5]<FWHM_threshold)&(data_S[5]>0)&(data_S[6]<sky_threshold)).T[0]

        

        dataA_sort = np.array([data_A[0][data_A_index],data_A[3][data_A_index],data_A[4][data_A_index]])
        dataC_sort = np.array([data_C[0][data_C_index],data_C[3][data_C_index],data_C[4][data_C_index]])
        dataS_sort = np.array([data_S[0][data_S_index],data_S[3][data_S_index],data_S[4][data_S_index]])

        data_prereturn = np.array([time,mag,1/errorbar])

        data_prereturn = killnan(data_prereturn)
        data_prereturn = killinf(data_prereturn)

        return np.array(KMT_official_args),killnan(np.array([data_prereturn[0],data_prereturn[1],1/data_prereturn[2]])), killnan(dataA_sort), killnan(dataC_sort), killnan(dataS_sort),bound_cut 
    else: 
        time_select_index = np.argwhere((time>t_0_KMT+cutratio[1]*t_E_KMT)|(time<t_0_KMT+cutratio[0]*t_E_KMT)).T[0]
        bound_cut = [time_select_index[0],time_select_index[-1]]
        time = time[time_select_index]
        mag = mag[time_select_index]
        errorbar = errorbar[time_select_index]
        # print(len(time))
        data_A_index = np.argwhere((data_A[4]<err_threshold)&((data_A[0]>t_0_KMT+cutratio[1]*t_E_KMT)|(data_A[0]<t_0_KMT+cutratio[0]*t_E_KMT))&(data_A[5]<FWHM_threshold)&(data_A[5]>0)&(data_A[6]<sky_threshold)).T[0]
        data_C_index = np.argwhere((data_C[4]<err_threshold)&((data_C[0]>t_0_KMT+cutratio[1]*t_E_KMT)|(data_C[0]<t_0_KMT+cutratio[0]*t_E_KMT))&(data_C[5]<FWHM_threshold)&(data_C[5]>0)&(data_C[6]<sky_threshold)).T[0]
        data_S_index = np.argwhere((data_S[4]<err_threshold)&((data_S[0]>t_0_KMT+cutratio[1]*t_E_KMT)|(data_S[0]<t_0_KMT+cutratio[0]*t_E_KMT))&(data_S[5]<FWHM_threshold)&(data_S[5]>0)&(data_S[6]<sky_threshold)).T[0]

        

        dataA_sort = np.array([data_A[0][data_A_index],data_A[3][data_A_index],data_A[4][data_A_index]])
        dataC_sort = np.array([data_C[0][data_C_index],data_C[3][data_C_index],data_C[4][data_C_index]])
        dataS_sort = np.array([data_S[0][data_S_index],data_S[3][data_S_index],data_S[4][data_S_index]])

        data_prereturn = np.array([time,mag,1/errorbar])

        data_prereturn = killnan(data_prereturn)
        data_prereturn = killinf(data_prereturn)

        return np.array(KMT_official_args),killnan(np.array([data_prereturn[0],data_prereturn[1],1/data_prereturn[2]])), killnan(dataA_sort), killnan(dataC_sort), killnan(dataS_sort),bound_cut

# Function of fitting
def magnitude_tran(A,m0,fb=0):
    return m0 - 2.5*np.log10((1-fb)*A+fb)

def mag_cal(t,tE,t0,u0,m0,fb=0):
    u = np.sqrt(((t-t0)/tE)**2+u0**2)
    A = (u**2+2)/(u*np.sqrt(u**2+4))
    return magnitude_tran(A,m0,fb)

def chi_square(single,binary,sigma):
    return np.sum(np.power((single-binary)/sigma,2))

def chi2_for_minimize(args,time,mag,sigma):
    return chi_square(mag_cal(time,*args),mag,sigma)

def doublefitting(time,mag,err,init_args):
    # initial_guess = [tE,t0,u0,m0,fb]
    tE,t0,u0,m0,fb = init_args
    print(init_args)

    mag_single_model_phy = mag_cal(time,*init_args)
    diff_phymodel = np.abs(mag-mag_single_model_phy)
    mean_diff_phymodel = np.mean(diff_phymodel)

    choose_list = (diff_phymodel <= 1*mean_diff_phymodel)

    print(len(choose_list))
    print(np.sum(choose_list))

    # Minimize using extend fitting
    ## [u_0, rho, q, s, alpha, t_E, basis_m, t_0]
    result = op.minimize(chi2_for_minimize, x0=init_args,args=(time[choose_list],mag[choose_list],err[choose_list]), method='Nelder-Mead')    

    # print("Fitting was successful? {:}".format(result.success))
    # print("Function evaluations: {:}".format(result.nfev))
    if isinstance(result.fun, np.ndarray):
        if result.fun.ndim == 0:
            result_fun = float(result.fun)
        else:
            result_fun = result.fun[0]
    else:
        result_fun = result.fun

    minimize_args = result.x.tolist()

    return minimize_args, mag_cal(time,*minimize_args)




'''
def mag_ori(A):
    return -2.5*np.log10(A)


def KMTmag_fit(t,t0,tE,u0,m0,fb):
    u = np.sqrt(((t-t0)/tE)**2+u0**2)
    A = (u**2+2)/(u*np.sqrt(u**2+4))
    return m0-2.5*np.log10((1-fb)*A+fb)

def chi2_for_minimize(args,time,mag,sigma):
    return chi_square(KMTmag_fit(time,*args),mag,sigma)

def fitKMTdata(data, init_args):# args: t0,tE,u0,m0,fb
     # first fitting
    time,mag,err = data
    initial_guess = init_args.copy()
    result = op.minimize(chi2_for_minimize, x0=initial_guess,args=(time,mag,err), method='Nelder-Mead')
    args_minimize_fortest = result.x.tolist()
    return np.array(args_minimize_fortest)


def alignKMTdata(datas,datas_fit,init_args):
    time_align = []
    magnification_align = []
    err_align = []

    m0_list = []

    for i,data in enumerate(datas):
        time,mag,err = np.array(data)
        args_fit = fitKMTdata(datas_fit[i],init_args)
        t0,tE,u0,m0,fb = args_fit

        magnification = (10**((m0-mag)/2.5)-fb)/(1-fb)

        time_align = np.append(time_align, time)
        magnification_align = np.append(magnification_align,magnification)
        err_align = np.append(err_align,err)
        m0_list.append(10**(-m0/2.5))

    m0_mean = -2.5*np.log10(np.mean(m0_list))

    mag_align = m0_mean-2.5*np.log10(magnification_align)

    order_align = np.argsort(time_align)

    return time_align[order_align], mag_align[order_align], err_align[order_align]
'''