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
import multiprocessing as mp

import netmodule.netGRUiden as lcnet


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="4"

trainortest = 1 # 0:test, 1:train
fullorparttest = 0 # 0: part testfig 1: full testfig

name_group = "20to25"
# prepare

# reload
reload = 0
preload_Netmodel = "GRUresnet_iden_res_"+name_group+".pkl"
path_params = "/scratch/zerui603/netparams/"
num_process = 16

# initialize GPU
use_gpu = torch.cuda.is_available()

print(os.cpu_count())
# device_ids = [1,2,3,4]

# torch.cuda.set_device("cuda:4,5")

torch.backends.cudnn.benchmark = True

# define parameters
## number of points

## size of trainingset library
size_train = 700000
## size of validationset library
size_val = 70000

## batch size and epoch
batch_size_train = 35000
batch_size_val =10000
n_epochs = 30
learning_rate = 8e-6
stepsize = 6
gamma_0 = 0.8
momentum = 0.5

## path of trainingset and validationset

rootdir = "/scratch/zerui603/KMT_simu_lowratio/qseries/"+name_group+"/"
rootval = "/scratch/zerui603/KMT_simu_lowratio/qseries/"+name_group+"test/"
rootdraw = "/home/zerui603/MDN_lc/iden_1D/testfig/"
fullrootdraw = "/scratch/zerui603/figtest_iden/"+name_group+"/"

## arguments for training
num_test = 70000
batch_test = 12000

def signlog(x):
    return np.log10(np.abs(x))

class testdraw:
    def __init__(self,filelist,datadir,filelabel,fullrootdraw):
        self.filelist = filelist
        self.datadir = datadir
    
        self.size_filelist = len(filelist)
        self.fullrootdraw = fullrootdraw
        self.label_list = ["bb","bs","sb","ss"]
        self.filelabel = filelabel

    def draw_testfig(self,index_fulltest):
        data = list(np.load(self.datadir+str(self.filelist[index_fulltest])+".npy", allow_pickle=True))
        labels = list(np.array(data[0],dtype=np.float64))

        ## [u_0, rho, q, s, alpha, t_E, basis_m, t_0, chi^2, label]
        ## [times, dtimes, lc_noi, sigma, lc_nonoi, args_minimize, lc_fit_minimize, chi_array]

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
            plt.suptitle(r"$\log_{10} q$=%.3f,$\log_{10} s=$%.3f,$u_0=$%.3f,$\alpha=$%.1f,$\Delta \chi^2$:%.3f,index=%d"%(np.log10(labels[2]),np.log10(labels[3]),labels[0],labels[4],labels[-2],self.filelist[index_fulltest],))
        else:
            plt.suptitle("$\Delta \chi^2$:%.3f,index=%d"%(labels[-2],self.filelist[index_fulltest],))

        plt.savefig(self.fullrootdraw+self.label_list[self.filelabel]+"/"+str(index_fulltest)+".png")
        plt.close()

        plt.figure(figsize=[12,18])
        plt.subplot(211)
        plt.errorbar(time,lc,yerr=sig,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,zorder=0)
        plt.scatter(time,lc,s=4,c="r",label = "lightcurve")
        
        plt.xlabel("time/HJD")
        plt.ylabel("magnitude")
        plt.ylim(mag_min_lim,mag_max_lim)
        plt.legend()
        plt.gca().invert_yaxis()
        plt.subplot(212)
        plt.plot(time,lc)
        plt.xlabel("time/HJD")
        plt.ylabel("magnitude")
        plt.ylim(mag_min_lim,mag_max_lim)
        
        plt.gca().invert_yaxis()

        if labels[-1] == 1:
            plt.suptitle(r"$\log_{10} q$=%.3f,$\log_{10} s=$%.3f,$u_0=$%.3f,$\alpha=$%.1f,$\Delta \chi^2$:%.3f,index=%d"%(np.log10(labels[2]),np.log10(labels[3]),labels[0],labels[4],labels[-2],self.filelist[index_fulltest],))
        else:
            plt.suptitle("$\Delta \chi^2$:%.3f,index=%d"%(labels[-2],self.filelist[index_fulltest],))

        plt.savefig(self.fullrootdraw+self.label_list[self.filelabel]+"_onlyfig/"+str(index_fulltest)+".png")
        plt.close()



def testnet(datadir=rootval,fullorparttest = fullorparttest):
    # initialize model
    network = lcnet.ResNet()
    criterion = nn.BCELoss()
    if use_gpu:
        # network = network.cuda()
        criterion = criterion.cuda()
        network = nn.DataParallel(network).cuda()

    network.load_state_dict(torch.load(path_params+preload_Netmodel))

    bchi_s_pre = []
    bchi_s_act = []
    blabel_pre = []
    blabel_act = []
    schi_s_pre = []
    schi_s_act = []
    slabel_pre = []
    slabel_act = []

    # file_actual/predicted
    file_bb = []
    file_bs = []
    file_sb = []
    file_ss = []

    num_batch = num_test//batch_test

    if num_test%batch_test != 0:
        num_batch += 1

    for index_batch in range(num_batch):

        input_batch = []
        label_batch = []
        chi_s_batch = []
        file_batch = []


        for index in range(index_batch*batch_test,np.min([(index_batch+1)*batch_test,num_test])):
            lc_data,args = lcnet.default_loader_fortest(data_root = datadir,posi_lc = index)
            input_batch.append(lc_data)

            file_batch.append(index)

            dchi_s = args[-2]
            bslabel = args[-1]

            
            if (bslabel>0.5)&(dchi_s < 20):
                bslabel = 0
            
            label_batch.append(bslabel)
            chi_s_batch.append(dchi_s)
        
        input_batch = torch.from_numpy(np.array(input_batch)).float()
        if use_gpu:
            input_batch = input_batch.cuda()
        
        network.eval()
        output_batch = network(input_batch).detach().cpu().numpy()


        bspre_batch = np.around(output_batch.T[0])
        label_batch = np.array(label_batch).astype(np.int)

        file_batch = np.array(file_batch)

        file_bb_batch = file_batch[(label_batch>0.5)&(bspre_batch>0.5)]
        file_bs_batch = file_batch[(label_batch>0.5)&(bspre_batch<0.5)]
        file_sb_batch = file_batch[(label_batch<0.5)&(bspre_batch>0.5)]
        file_ss_batch = file_batch[(label_batch<0.5)&(bspre_batch<0.5)] 

        if index_batch == 0:
            bchi_s_pre = np.array(chi_s_batch)[np.argwhere(bspre_batch==1).T[0]].copy()
            bchi_s_act = np.array(chi_s_batch)[np.argwhere(label_batch==1).T[0]].copy()
            schi_s_pre = np.array(chi_s_batch)[np.argwhere(bspre_batch==0).T[0]].copy()
            schi_s_act = np.array(chi_s_batch)[np.argwhere(label_batch==0).T[0]].copy()

            file_bb = file_bb_batch.copy()
            file_bs = file_bs_batch.copy()
            file_sb = file_sb_batch.copy()
            file_ss = file_ss_batch.copy()

        else:
            bchi_s_pre = np.append(bchi_s_pre, np.array(chi_s_batch)[np.argwhere(bspre_batch==1).T[0]] )
            bchi_s_act = np.append(bchi_s_act, np.array(chi_s_batch)[np.argwhere(label_batch==1).T[0]] )
            schi_s_pre = np.append(schi_s_pre, np.array(chi_s_batch)[np.argwhere(bspre_batch==0).T[0]] )
            schi_s_act = np.append(schi_s_act, np.array(chi_s_batch)[np.argwhere(label_batch==0).T[0]] )

            file_bb = np.append(file_bb, file_bb_batch)
            file_bs = np.append(file_bs, file_bs_batch)
            file_sb = np.append(file_sb, file_sb_batch)
            file_ss = np.append(file_ss, file_ss_batch)
    print(len(bchi_s_act))
    print(len(schi_s_act))
    print(len(bchi_s_pre))
    print(len(schi_s_pre))
    plt.figure(figsize=[7,20])
    plt.subplot(511)
    plt.hist(signlog(bchi_s_act),bins=1000,label="actual binary",alpha=0.5)
    plt.hist(signlog(schi_s_act),bins=1000,label="actual single",alpha=0.5)
    plt.hist(signlog(bchi_s_pre),bins=1000,label="predicted binary",alpha=0.5)
    plt.hist(signlog(schi_s_pre),bins=1000,label="predicted single",alpha=0.5)

    plt.xlabel("$\log_{10} |\Delta \chi^2|$")
    plt.legend()
    plt.subplot(512)
    plt.hist(signlog(bchi_s_act),bins=1000,label="actual binary",alpha=0.5)
    plt.legend()
    plt.xlabel("$\log_{10} |\Delta \chi^2|$")
    plt.subplot(513)
    plt.hist(signlog(schi_s_act),bins=1000,label="actual single",alpha=0.5)
    plt.legend()
    plt.xlabel("$\log_{10} |\Delta \chi^2|$")
    plt.subplot(514)
    plt.hist(signlog(bchi_s_pre),bins=1000,label="predicted binary",alpha=0.5)
    plt.legend()
    plt.xlabel("$\log_{10} |\Delta \chi^2|$")
    plt.subplot(515)
    plt.hist(signlog(schi_s_pre),bins=1000,label="predicted single",alpha=0.5)
    plt.legend()
    plt.xlabel("$\log_{10} |\Delta \chi^2|$")
    plt.savefig("histbs_"+name_group+".png")
    plt.close()

    # test events

    testsize=100
    label_list = ["bb","bs","sb","ss"]
    filename_list = [file_bb,file_bs,file_sb,file_ss]

    print("actual/predicted: bb,bs,sb,ss",len(file_bb),len(file_bs),len(file_sb),len(file_ss))

    if fullorparttest == 0:
        for index_draw in range(len(label_list)):

            # I don't know whether this operation is legal or not
            filelist = filename_list[index_draw]
            for index_testfig in range(testsize):
                size_filelist = len(filelist)
                index_random = random.randint(0,size_filelist-1)
                data = list(np.load(datadir+str(filelist[index_random])+".npy", allow_pickle=True))
                
                labels = list(np.array(data[0],dtype=np.float64))

                ## [u_0, rho, q, s, alpha, t_E, basis_m, t_0, chi^2, label]
                ## [times, dtimes, lc_noi, sigma, lc_nonoi, args_minimize, lc_fit_minimize, chi_array]

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
                    plt.suptitle(r"$\log_{10} q$=%.3f,$\log_{10} s=$%.3f,$u_0=$%.3f,$\alpha=$%.1f,$\Delta \chi^2$:%.3f,index=%d"%(np.log10(labels[2]),np.log10(labels[3]),labels[0],labels[4],labels[-2],index_random,))
                else:
                    plt.suptitle("$\Delta \chi^2$:%.3f,index=%d"%(labels[-2],index_random,))

                plt.savefig(rootdraw+label_list[index_draw]+"/"+str(index_testfig)+"_"+name_group+".png")
                plt.close()

        return 0
    elif fullorparttest == 1:
        return filename_list

def training():
    # Loading datas
    trainingsdata = lcnet.Mydataset(n_lc=size_train,data_root=rootdir,judge_train=0)
    trainset = lcnet.DataLoaderX(trainingsdata, batch_size=batch_size_train,shuffle=True,num_workers=16,pin_memory=True)

    valdata = lcnet.Mydataset(n_lc=size_val,data_root=rootval,judge_train=0)
    valset = lcnet.DataLoaderX(valdata, batch_size=batch_size_val,shuffle=True,num_workers=16,pin_memory=True)

    # initialize model
    network = lcnet.ResNet()
    criterion = nn.BCELoss()
    if use_gpu:
        # network = network.cuda()
        criterion = criterion.cuda()
        network = nn.DataParallel(network).cuda()
    if reload == 1:
        network.load_state_dict(torch.load(path_params+preload_Netmodel))

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=stepsize,gamma=gamma_0)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True, threshold=0.05, threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-7)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1)
    # Training

    loss_figure = []
    val_loss_figure = []
    val_correct_list = []

    cor_matrix = [[],[],[],[]]

    for epoch in range(n_epochs):
        running_loss = 0.0
        epoch_rs = 0
        sam = 0
        network.train()
        print("start training",datetime.datetime.now())
        for (i,data) in enumerate(trainset):
            # print("start loading data",datetime.datetime.now())
            inputs, labels = data
            inputs = inputs.float()
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            # print("finish loading data",datetime.datetime.now())
            optimizer.zero_grad()
            outputs = network(inputs)
            outputs = outputs.double()
            
            # print(np.max(outputs.detach().cpu().numpy()),np.min(outputs.detach().cpu().numpy()))
            # print(np.max(labels.detach().cpu().numpy()),np.min(labels.detach().cpu().numpy()))
            
            loss = criterion(outputs,labels)
            # print(labels)
        
            loss.backward()
            optimizer.step()
            # print("finish calculating",datetime.datetime.now())
            epoch_rs = epoch_rs + loss.detach().item()
            
            if sam%5 == 0:
                print("Epoch:[", epoch + 1, sam, "] loss:", loss.item(),str(datetime.datetime.now()))
                
            sam = sam+1
        
        
        loss_figure.append(epoch_rs/sam)
        print("Training_Epoch:[", epoch + 1, "] Training_loss:", epoch_rs/sam,str(datetime.datetime.now()))
        print("learning rate: ",optimizer.state_dict()['param_groups'][0]['lr'])

        if (epoch+1)%5 == 0:
            torch.save(network.state_dict(),path_params+preload_Netmodel)
            print("netparams have been saved once")
        if (epoch+1)%20 == 0:
            testnet(fullorparttest = 0)
            print("Examination have been executed once")

        gc.collect()
        

        val_epoch_rs = 0
        val_sam = 0
        val_correct = 0
        val_cor_00 = 0
        val_cor_01 = 0
        val_cor_10 = 0
        val_cor_11 = 0



        network.eval()
        with torch.no_grad():
            for j,valdata in enumerate(valset):
                val_inputs, val_labels = valdata
                val_inputs = val_inputs.float()
                if use_gpu:
                    val_inputs = val_inputs.cuda()
                    val_labels = val_labels.cuda()
                optimizer.zero_grad()
                val_outputs = network(val_inputs)
                val_outputs = val_outputs.double()
                loss = criterion(val_outputs,val_labels)
                val_sam = val_sam + 1
                val_epoch_rs = val_epoch_rs + loss.item()
                print("val:",val_sam,loss.item())

                val_outputs = np.around(val_outputs.cpu().detach().numpy())
                val_labels = val_labels.cpu().detach().numpy()
                # print(val_labels.shape)
                correct_num = np.sum(val_labels*val_outputs)
                ## 
                labels_0 = val_labels.T[0]
                labels_1 = val_labels.T[1]
                output_0 = val_outputs.T[0]
                output_1 = val_outputs.T[1]

                val_correct += correct_num
                val_cor_00 += np.sum(labels_0*output_0)
                val_cor_01 += np.sum(labels_0*output_1)
                val_cor_10 += np.sum(labels_1*output_0)
                val_cor_11 += np.sum(labels_1*output_1)



        val_loss_figure.append(val_epoch_rs/val_sam)
        val_correct_list.append(val_correct/size_val)

        scheduler.step()


        print("val_Epoch:[", epoch + 1, "] val_loss:", val_epoch_rs/val_sam,str(datetime.datetime.now()))
        print("Correct valset: ",val_correct,"/",size_val)
        print("actual/predicted: bb,bs,sb,ss",val_cor_00,val_cor_01,val_cor_10,val_cor_11)
        print("actual/predicted rate: bb,bs,sb,ss",val_cor_00/(val_cor_00+val_cor_01),val_cor_01/(val_cor_00+val_cor_01),val_cor_10/(val_cor_10+val_cor_11),val_cor_11/(val_cor_10+val_cor_11))
        cor_matrix[0].append(val_cor_00)
        cor_matrix[1].append(val_cor_01)
        cor_matrix[2].append(val_cor_10)
        cor_matrix[3].append(val_cor_11)


        plt.figure(figsize=(18,24))
        plt.subplot(411)
        x = np.linspace(1,epoch+1,len(loss_figure))
        plt.plot(x, loss_figure,label = "training loss log-likehood")
        plt.plot(x, val_loss_figure,label = "val loss log-likehood")
        plt.title("loss-epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss BCELoss")
        plt.legend()

        plt.subplot(412)
        plt.plot(x, val_correct_list,label="accuracy")
        plt.title("Accuracy")
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")

        plt.subplot(413)
        plt.plot(x, cor_matrix[0],label="output:binary,label:binary")
        plt.plot(x, cor_matrix[1],label="output:single,label:binary")
        plt.plot(x, cor_matrix[2],label="output:binary,label:single")
        plt.plot(x, cor_matrix[3],label="output:single,label:single")
        plt.xlabel("epoch")
        plt.ylabel("Number")
        plt.legend()

        plt.subplot(414)
        plt.plot(x, cor_matrix[0]/(np.array(cor_matrix[0])+np.array(cor_matrix[1])),label="output:binary,label:binary")
        plt.plot(x, cor_matrix[1]/(np.array(cor_matrix[0])+np.array(cor_matrix[1])),label="output:single,label:binary")
        plt.plot(x, cor_matrix[2]/(np.array(cor_matrix[2])+np.array(cor_matrix[3])),label="output:binary,label:single")
        plt.plot(x, cor_matrix[3]/(np.array(cor_matrix[2])+np.array(cor_matrix[3])),label="output:single,label:single")
        plt.xlabel("epoch")
        plt.ylabel("rate")
        plt.legend()
        
        plt.savefig("loss_accuracy_GRU1d_"+name_group+".png")
        plt.close()


    torch.save(network.state_dict(),path_params+preload_Netmodel)

if __name__=="__main__":
    if trainortest == 1:
        training()
    else:
        filename_list = testnet(fullorparttest = fullorparttest)

        if fullorparttest == 1:
            for index_draw in range(0,1):
                filelist = filename_list[index_draw]
                size_filelist = len(filelist)
                test_file_draw = testdraw(filelist = filelist,datadir=rootval,filelabel=index_draw,fullrootdraw = fullrootdraw)
                with mp.Pool(num_process) as p:
                    p.map(test_file_draw.draw_testfig, range(size_filelist))
                    p.close()
                    p.join()
