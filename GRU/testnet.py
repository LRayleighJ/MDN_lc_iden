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
import datetime
import MulensModel as mm
import scipy.signal as signal
import scipy.fft
# from torchviz import make_dot 

import netmodule.netGRU as lcnet
import datamodule.datahist as dhist

# initialize preload netparams
reload = 1
preload_Netmodel = "/home/zerui603/MDN_lc/GRUresnet_params_4args.pkl"

# initialize the storage path
rootval = "/scratch/zerui603/KMTsimudata/test_smooth/"
samplepath = "/home/zerui603/MDN_lc/GRU/testfig/"

test_deg_index = 3

# initialize GPU
use_gpu = torch.cuda.is_available()
N_gpu = torch.cuda.device_count()
device_ids = range(N_gpu)
torch.backends.cudnn.benchmark = True

# initialize network
network = lcnet.ResNet()
criterion = lcnet.Loss_fn()
if use_gpu:
    network = network.cuda()
    criterion = criterion.cuda()
    network = nn.DataParallel(network)
if reload == 1:
    network.load_state_dict(torch.load(preload_Netmodel))


# test single lc
# initialize the size of valsets
test_size = 1743
total_size = 1743

resample_size = 1743

index_list = list(range(resample_size))

test_cadence = 20

# initialize the range of args-space
## Attention: the args that input into network are multiplied by 5

range_lgq = 2
range_lgs = 0.6
range_ux = 1
range_uy = 1

range_lgq_forsolve = 4
range_lgs_forsolve = 2
range_ux_forsolve = 2
range_uy_forsolve = 2

# get actual-predicted figure
act_args = [[],[],[],[]]
pre_args = [[],[],[],[]]
prob_args = [[],[],[],[]]
# data loader
def lightcurve_loader(data_root,posi_lc,judge_train=0):
    # [u_0, rho, q, s, alpha, t_E]
    # datadir = pd.DataFrame(np.load(dataroot+str(posi_lc)+".npy", allow_pickle=True))
    datadir = list(np.load(data_root+str(posi_lc+1000000*judge_train)+".npy", allow_pickle=True))
    
    labels = np.array(datadir[0],dtype=np.float64)

    lc_mag = np.array(datadir[3],dtype=np.float64)
    lc_mag = np.mean(np.sort(lc_mag)[-50:])-np.array(lc_mag)
    lc_mag = lc_mag.reshape((1000,1))
    lc_time = np.array(datadir[1],dtype=np.float64)
    lc_time = (lc_time-lc_time[0])/(lc_time[-1]-lc_time[0])
    lc_time = lc_time.reshape((1000,1))
    lc_sig = np.array(datadir[4],dtype=np.float64).reshape((1000,1))
    lc_sig = lc_sig*100

    data_input = np.concatenate((lc_mag,lc_time,lc_sig),axis=1)

    # print(data_input.shape)

    lg_q = np.log10(labels[2])
    lg_s = np.log10(labels[3])
    alpha = labels[4]
    u0 = labels[0]

    q_label = lg_q/4
    s_label = (lg_s-np.log10(0.3))/(np.log10(3)-np.log10(0.3))
    # alpha_label = alpha/360
    # u0_label = u0
    ux_label = (u0*np.cos(np.pi/180*alpha)+1)/2
    uy_label = (u0*np.sin(np.pi/180*alpha)+1)/2
    

    label = np.array([-1.*float(q_label),float(s_label),float(ux_label),float(uy_label)]).astype(np.float64)
    lc_data = np.array([data_input])

    return lc_data


for index_sample in range(test_size):
    data = list(np.load(rootval+str(index_list[index_sample])+".npy", allow_pickle=True))
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

    # raise RuntimeError("Test")

    # Order of args: [u_0, rho, q, s, alpha, t_E]
    lg_q = np.log10(labels[2])
    lg_s = np.log10(labels[3])
    ux = labels[0]*np.cos(np.pi/180*labels[4])
    uy = labels[0]*np.sin(np.pi/180*labels[4])
    realargs = [lg_q,lg_s,ux,uy]

    # origin histed lightcurve
    time_ref = np.linspace(lc_time[0],lc_time[-1],1000)
    lc_ref = dhist.lc_hist(lc_withnoi,lc_time,lc_sig,num=1000)

    data_input = lightcurve_loader(data_root = rootval, posi_lc = index_sample)
    print(data_input.shape)
    inputs = torch.from_numpy(data_input).float()
    
    if use_gpu:
        inputs = inputs.cuda()
    network.eval()
    # Order of outputs: lgq,lgs,ux,uy
    # Silly code......
    pi1,pi2,pi3,pi4,mu1,mu2,mu3,mu4,sigma1,sigma2,sigma3,sigma4 = network(inputs)
    pi1 = pi1.detach().cpu().numpy()[0]
    pi2 = pi2.detach().cpu().numpy()[0]
    pi3 = pi3.detach().cpu().numpy()[0]
    pi4 = pi4.detach().cpu().numpy()[0]
    pis = np.exp(np.array([pi1,pi2,pi3,pi4]))
    mu1 = mu1.detach().cpu().numpy()[0]
    mu2 = mu2.detach().cpu().numpy()[0]
    mu3 = mu3.detach().cpu().numpy()[0]
    mu4 = mu4.detach().cpu().numpy()[0]
    mus = np.array([mu1,mu2,mu3,mu4])
    sigma1 = sigma1.detach().cpu().numpy()[0]
    sigma2 = sigma2.detach().cpu().numpy()[0]
    sigma3 = sigma3.detach().cpu().numpy()[0]
    sigma4 = sigma4.detach().cpu().numpy()[0]
    sigmas = np.array([sigma1,sigma2,sigma3,sigma4])
    # calc the range of args
    lgq_domain = np.linspace(lg_q-range_lgq,lg_q+range_lgq,200)
    lgs_domain = np.linspace(-1*np.abs(lg_s)-range_lgs,np.abs(lg_s)+range_lgs,200)
    ux_domain = np.linspace(ux-range_ux,ux+range_ux,200)
    uy_domain = np.linspace(uy-range_uy,uy+range_uy,200)

    args_domains = np.array([lgq_domain,lgs_domain,ux_domain,uy_domain])

    # domain for solve

    lgq_domain_fs = np.linspace(lg_q-range_lgq_forsolve,lg_q+range_lgq_forsolve,1000)
    lgs_domain_fs = np.linspace(-1*np.abs(lg_s)-range_lgs_forsolve,np.abs(lg_s)+range_lgs_forsolve,1000)
    ux_domain_fs = np.linspace(ux-range_ux_forsolve,ux+range_ux_forsolve,1000)
    uy_domain_fs = np.linspace(uy-range_uy_forsolve,uy+range_uy_forsolve,1000)

    args_domains_fs = np.array([lgq_domain_fs,lgs_domain_fs,ux_domain_fs,uy_domain_fs])


    axislabels = [r"$\log_{10} q$",r"$\log_{10} s$",r"$u_0 \cos \alpha$",r"$u_0 \sin \alpha$"]

    # plot actual-predicted figure
    # [lgq,lgs,ux,uy]

    test_deg_args = [[],[],[],[]]
    

    for index_paf in range(4):
        net_domain_fs = dhist.tran_universal(args_domains_fs[index_paf],index_paf)
        density_domain_fs = dhist.multi_gaussian_prob(pis[index_paf],mus[index_paf],sigmas[index_paf])(net_domain_fs)
        solution,prob_value = dhist.get_solution(args_domains_fs[index_paf],density_domain_fs*10000)
        prob_index = np.argsort(prob_value)
        for index_solution in range(len(solution)):
            act_args[index_paf].append(realargs[index_paf])
            pre_args[index_paf].append(solution[index_solution])
            prob_args[index_paf].append(prob_value[index_solution])
        if index_paf == test_deg_index:
            test_deg_args[index_paf].append(solution[prob_index[-1]])
            try:
                test_deg_args[index_paf].append(solution[prob_index[-2]])
            except:
                continue
        else:
            test_deg_args[index_paf].append(solution[prob_index[-1]])
        # prob_value = np.array(prob_value)/np.max(prob_value)
        

    # Plot density figure
    if index_sample%test_cadence == 0:
        mat_id = (np.array(range(3*3))+1).reshape((3,3)).T
        plot_base = 330

        print("lc: ",index_sample,test_deg_args)

        # test_arguments init
        # Order of args: [u_0, rho, q, s, alpha, t_E]
        q_t = 0
        s1_t = 0
        s2_t = 0
        u0_t = 0
        u01_t = 0
        u02_t = 0
        alpha_t = 0
        alpha1_t = 0
        alpha2_t = 0
        tE_t = labels[-1]
        t0_t = 0
        ux1_t=0
        ux2_t=0
        uy_t=0

        q_t = 10**(test_deg_args[0][0])
        s1_t = 10**(test_deg_args[1][0])
        # ux1_t = test_deg_args[test_deg_index][0]
        u01_t = np.sqrt(test_deg_args[2][0]**2+test_deg_args[3][0]**2)
        alpha1_t = dhist.cal_alpha(test_deg_args[2][0],test_deg_args[3][0])
        if len(test_deg_args[test_deg_index]) >1:
            # ux2_t = 10**(test_deg_args[test_deg_index][1])
            u02_t = np.sqrt(test_deg_args[2][0]**2+test_deg_args[3][1]**2)
            alpha2_t = dhist.cal_alpha(test_deg_args[2][0],test_deg_args[3][1])

        re_model_1 = mm.Model({'t_0': t0_t, 'u_0': u01_t,'t_E': tE_t, 'rho': labels[1], 'q': q_t, 's': s1_t,'alpha': alpha1_t})
        re_model_1.set_default_magnification_method("VBBL")
        re_mag_1 = re_model_1.magnification(time_ref)
        re_mag_1 = dhist.magnitude_tran(re_mag_1)
        re_mag_1 = np.mean(np.sort(re_mag_1)[-50:])-np.array(re_mag_1)
        if len(test_deg_args[test_deg_index]) > 1:
            re_model_2 = mm.Model({'t_0': t0_t, 'u_0': u02_t,'t_E': tE_t, 'rho': labels[1], 'q': q_t, 's': s1_t,'alpha': alpha2_t})
            re_model_2.set_default_magnification_method("VBBL")
            re_mag_2 = re_model_2.magnification(time_ref)
            re_mag_2 = dhist.magnitude_tran(re_mag_2)
            re_mag_2 = np.mean(np.sort(re_mag_2)[-50:])-np.array(re_mag_2)


        plt.figure(figsize=(24,24)) 
        for index_i in range(3):
            for index_j in range(index_i+1,4):
                x_domain = dhist.tran_universal(args_domains[index_i],index_i)
                y_domain = dhist.tran_universal(args_domains[index_j],index_j)
                X,Y = np.meshgrid(x_domain,y_domain)
                x_prob = dhist.multi_gaussian_prob(pis[index_i],mus[index_i],sigmas[index_i])(X)
                y_prob = dhist.multi_gaussian_prob(pis[index_j],mus[index_j],sigmas[index_j])(Y)
                xy_prob = x_prob*y_prob
                plt.subplot(plot_base+mat_id[index_i][index_j-1])
                h = plt.contourf(args_domains[index_i], args_domains[index_j], xy_prob,cmap="Blues")
                plt.scatter([realargs[index_i]],[realargs[index_j]],s=120,c="r",marker="x")
                plt.xlabel(axislabels[index_i],fontsize=30)
                plt.ylabel(axislabels[index_j],fontsize=30)
                c = plt.colorbar(h)
                # plt.axis("equal")
                plt.grid()
        plt.suptitle(r"$\log_{10} q$=%.3f,$\log_{10} s=$%.3f,$u_0=$%.3f,$\alpha=$%.1f"%(lg_q,lg_s,labels[0],labels[4]),fontsize=40)
        plt.savefig(samplepath+str(index_sample)+".png")
        plt.close()
        
        


        plt.figure(figsize=(20,9))
        plt.subplot(121)
        plt.errorbar(lc_time,lc_withnoi,yerr=lc_sig,fmt='o',capsize=2,elinewidth=1,ms=3,alpha=0.7,zorder=0)
        plt.scatter(time_ref,lc_ref,c="r",s=5,alpha=0.5,zorder=1,label="input data")
        plt.scatter(time_ref,re_mag_1,s=5,c="g",alpha=0.5,zorder=1,label="predicted lightcurve")
        # if len(test_deg_args[test_deg_index]) == 2:
        #     plt.scatter(time_ref,re_mag_2,c="y",s=5,alpha=0.5,zorder=1,label="degeneracy lightcurve")
        plt.legend()
        # plt.title("lg q=%.3f,lg s=%.3f,u0=%.3f,alpha=%.1f"%(lg_q,lg_s,labels[0],labels[4]))
        
        plt.subplot(122)
        ## trajectory
        ## Order of args: [u_0, rho, q, s, alpha, t_E]
        bl_model = mm.Model({'t_0': 0, 'u_0': labels[0],'t_E': labels[5], 'rho': labels[1], 'q': labels[2], 's': labels[3],'alpha': labels[4]})
        bl_model.set_default_magnification_method("VBBL")
        
        
        caustic = mm.Caustics(s=labels[3], q=labels[2])
        X_caustic,Y_caustic = caustic.get_caustics(n_points=2000)

        trace_x = -np.sin(labels[4]*np.pi/180)*labels[0]+lc_time/labels[5]*np.cos(labels[4]*np.pi/180)
        trace_y = np.cos(labels[4]*np.pi/180)*labels[0]+lc_time/labels[5]*np.sin(labels[4]*np.pi/180)
        
        plt.scatter(X_caustic,Y_caustic,s=1,c="b")
        plt.plot(trace_x,trace_y,c="g")
        plt.xlabel(r"$\theta_x$")
        plt.ylabel(r"$\theta_y$")
        plt.axis("scaled")
        plt.grid()
        plt.suptitle(r"$\log_{10} q$=%.3f,$\log_{10} s=$%.3f,$u_0=$%.3f,$\alpha=$%.1f"%(lg_q,lg_s,labels[0],labels[4]),fontsize=30)
        plt.savefig(samplepath+str(index_sample)+"_lc.png")
        plt.close()

    # print("lightcurve %d has finished"%(index_sample,),datetime.datetime.now())

# a-p figure
axislabelnew=[r"$\log_{10} q$",r"$\log_{10} s$",r"$u_0 \cos \alpha$",r"$u_0 \sin \alpha$"]
plt.figure(figsize=[22.44,18])
for i in range(1,5):
    plt.subplot(220+i)
    alpha = np.array(prob_args[i-1])/10000

    act_plot = act_args[i-1]
    pre_plot = pre_args[i-1]


    plt.scatter(act_plot,pre_plot,s=20,zorder=0,alpha=0.8,c=alpha,cmap="Blues")
    if i <= 2:
        scaleline = np.linspace(min(act_args[i-1]),max(act_args[i-1]),200)
        plt.plot(scaleline,scaleline,zorder=1,alpha=0.5,c="r",linestyle="--")
        if i == 2:
            plt.plot(scaleline,-1*scaleline,zorder=1,alpha=0.25,c="b",linestyle="--")
        plt.xlabel("actual "+axislabelnew[i-1])
        plt.ylabel("predicted "+axislabelnew[i-1])
        plt.xlim((min(act_args[i-1]),max(act_args[i-1])))
        plt.ylim((min(act_args[i-1]),max(act_args[i-1])))
        plt.axis([min(act_args[i-1]),max(act_args[i-1]),min(act_args[i-1]),max(act_args[i-1])])
        plt.colorbar()
    else:
        scaleline = np.linspace(-1.2,1.2,200)
        plt.plot(scaleline,scaleline,zorder=1,alpha=0.5,c="r",linestyle="--")
        plt.plot(scaleline,-1*scaleline,zorder=1,alpha=0.25,c="b",linestyle="--")
        plt.xlabel("actual "+axislabelnew[i-1])
        plt.ylabel("predicted "+axislabelnew[i-1])
        plt.xlim((-1.2,1.2))
        plt.ylim((-1.2,1.2))
        plt.axis([-1.2,1.2,-1.2,1.2])
        plt.colorbar()
        

plt.savefig("pre-act.png")
plt.close()