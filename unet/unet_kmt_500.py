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
import sys
import imageio
import traceback

import datamodule.dm as dm
import netmodule.unetforkmt_500 as lcnet
import datamodule.loaddata as loaddata

def chis(x1,x2,sig,weight=1):
    return np.sum((np.array(x1)-np.array(x2))**2/np.array(sig)**2*weight)

def chis_array(x1,x2,sig,weight=1):
    return (np.array(x1)-np.array(x2))**2/np.array(sig)**2*weight

def log10abs(x):
    return np.log10(np.abs(x))

def map01(x):
    return (x-np.min(x))/(np.ptp(x))

        

testsize=10000

# reload
reload = 0
preload_Netmodel = "GRU_unet_500.pkl"
path_params = "/scratch/zerui603/netparams/"
num_process = 16

# initialize GPU
use_gpu = torch.cuda.is_available()
print("GPU:", use_gpu)

if use_gpu:
    pass
else:
    print("GPU is unavailable")
    exit()

print(os.cpu_count())

# device_ids = [1,2,3,4]

# torch.cuda.set_device("cuda:4,5")

torch.backends.cudnn.benchmark = True

# define parameters
## number of points

## size of trainingset library
size_train = 200000
## size of validationset library
size_val = 20000

## batch size and epoch
batch_size_train = 10000
batch_size_val =1000
n_epochs = 250
learning_rate = 4e-3
stepsize = 10# 7
gamma_0 = 0.85
momentum = 0.5

## path of trainingset and validationset

rootdir = "/scratch/zerui603/KMT_unet/extra_noise/training/"
rootval = "/scratch/zerui603/KMT_unet/extra_noise/val/"

# training

def training(paramsid,residual=False):
    # Loading datas
    trainingdata = lcnet.Mydataset(n_lc=size_train,data_root=rootdir,residual=residual)
    trainset = lcnet.DataLoaderX(trainingdata, batch_size=batch_size_train,shuffle=True,num_workers=num_process,pin_memory=True)

    valdata = lcnet.Mydataset(n_lc=size_val,data_root=rootval,residual=residual)
    valset = lcnet.DataLoaderX(valdata, batch_size=batch_size_val,shuffle=True,num_workers=num_process,pin_memory=True)

    # initialize model
    network = lcnet.Unet()

    weights = [1.0, 8.0]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # criterion = nn.CrossEntropyLoss()
    if use_gpu:
        # network = network.cuda()
        criterion = criterion.cuda()
        network = nn.DataParallel(network).cuda()
    if reload == 1:
        network.load_state_dict(torch.load(path_params+preload_Netmodel[:-4]+"_"+str(paramsid)+".pkl"))

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=stepsize,gamma=gamma_0)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True, threshold=0.05, threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-7)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1)
    # Training

    loss_figure = []
    val_loss_figure = []

    for epoch in range(n_epochs):
        running_loss = 0.0
        epoch_rs = 0
        sam = 0
        network.train()
        print("start training",datetime.datetime.now())
        for (i,data) in enumerate(trainset):
            
            inputs, labels = data
            inputs = inputs.float()
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()
            
            outputs = network(inputs)
            # outputs = outputs.double()
            
            loss = criterion(outputs,labels)
            # print(labels)
        
            loss.backward()
            optimizer.step()
            # print("finish calculating",datetime.datetime.now())
            epoch_rs = epoch_rs + loss.detach().item()
            
            if sam%1 == 0:
                print("Epoch:[", epoch + 1, sam, "] loss:", loss.item(),str(datetime.datetime.now()))
                
            sam = sam+1
        
        
        loss_figure.append(epoch_rs/sam)
        print("Training_Epoch:[", epoch + 1, "] Training_loss:", epoch_rs/sam,str(datetime.datetime.now()))
        print("learning rate: ",optimizer.state_dict()['param_groups'][0]['lr'])

        if (epoch+1)%10 == 0:
            torch.save(network.state_dict(),path_params+preload_Netmodel[:-4]+"_"+str(epoch+1)+".pkl")
            print("netparams have been saved once",epoch+1)

        gc.collect()

        val_epoch_rs = 0
        val_sam = 0

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
                # val_outputs = val_outputs.double()
                loss = criterion(val_outputs,val_labels)
                val_sam = val_sam + 1
                val_epoch_rs = val_epoch_rs + loss.item()
                print("val:",val_sam,loss.item())

        val_loss_figure.append(val_epoch_rs/val_sam)

        scheduler.step()


        print("val_Epoch:[", epoch + 1, "] val_loss:", val_epoch_rs/val_sam,str(datetime.datetime.now()))

        plt.figure()
        x = np.linspace(1,epoch+1,len(loss_figure))
        plt.plot(x, loss_figure,label = "training loss log-likelihood")
        plt.plot(x, val_loss_figure,label = "val loss log-likelihood")
        plt.title("loss-epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss BCELoss")
        plt.legend()
        
        plt.savefig("/home/zerui603/MDN_lc_iden/loss_accuracy_Unet_lowratio_500.png")
        plt.close()

    torch.save(network.state_dict(),path_params+preload_Netmodel)
    np.save("Unet_loss_lowratio_500.npy",np.array([loss_figure,val_loss_figure]))

def testfig(num_test,network,network_ref=None,thres=0.9999,thres_ref=0.9999,num_skip=0):
    network.eval()
    network_ref.eval()
    
    with torch.no_grad():
        for i in range(num_test):
            lc_data, lc_label, extra_noise_list = lcnet.loader_fortest(rootval,i,num_skip,residual=True)
            extra_noise_index, extra_noise = extra_noise_list
            label,time,lc_withnoi,err,lc_withoutnoi,lc_singlemodel = lc_label

            # predict
            input_batch = torch.from_numpy(np.array(lc_data)).float()
            if use_gpu:
                input_batch = input_batch.cuda()
            output_batch = network(input_batch).detach().cpu().numpy()[0][1]
            output_batch_ref = network_ref(input_batch).detach().cpu().numpy()[0][1]
            # print(output_batch.shape)

            lc_withnoi = np.array(lc_withnoi)
            extra_noise_index = np.array(extra_noise_index,dtype=np.int)
            extra_noise = np.array(extra_noise,dtype=np.float)

            lc_withnoi[extra_noise_index] += extra_noise

            s_point = lc_withnoi[label<0.5]
            s_time = time[label<0.5]
            b_point = lc_withnoi[label>0.5]
            b_time = time[label>0.5]

            s_point_pre = lc_withnoi[output_batch<thres]
            s_time_pre = time[output_batch<thres]
            b_point_pre = lc_withnoi[output_batch>thres]
            b_time_pre = time[output_batch>thres]

            s_point_pre_ref = lc_withnoi[output_batch_ref<thres_ref]
            s_time_pre_ref = time[output_batch_ref<thres_ref]
            b_point_pre_ref = lc_withnoi[output_batch_ref>thres_ref]
            b_time_pre_ref = time[output_batch_ref>thres_ref]


            mag_max_lim = np.mean(np.sort(lc_withoutnoi)[-25:])
            mag_min_lim = np.mean(np.sort(lc_withoutnoi)[:25])
            mag_max_lim += 0.3*(mag_max_lim-mag_min_lim)
            mag_min_lim -= 0.3*(mag_max_lim-mag_min_lim)

            plt.figure(figsize=(10,6))
            plt.scatter(time,lc_withnoi,s=20,alpha=0.5)
            #plt.scatter(s_time,s_point,s=10,alpha=0.5,label = "label no structure",c="blue")
            plt.scatter(b_time,b_point,s=40,alpha=0.5,label = "label with structure",c="green",marker="*")
            #plt.scatter(s_time_pre,s_point_pre,s=10,alpha=0.5,label = "predict no structure")
            plt.scatter(b_time_pre,b_point_pre,s=40,alpha=0.7,label = "predict with structure (residual)",c="red",marker="+")
            plt.scatter(b_time_pre,b_point_pre,s=40,alpha=0.7,label = "predict with structure (no residual)",c="orange",marker="x")
            plt.scatter(time[extra_noise_index],lc_withnoi[extra_noise_index],s=15,alpha=0.5,label = "extra noise",c="black")
            plt.plot(time,lc_withoutnoi,ls="--",label="binary model",c="green",alpha=0.3)
            plt.plot(time,lc_singlemodel,ls="--",label="single model",c="red",alpha=0.3)
            plt.xlabel("t",fontsize=16)
            plt.ylabel("Mag",fontsize=16)
            plt.legend()
            plt.ylim(mag_min_lim,mag_max_lim)
            plt.gca().invert_yaxis()
            
            plt.savefig("/home/zerui603/MDN_lc_iden/unet/testfig/"+str(np.int(i))+"_residual.png")
            plt.close()

def test_threshold(paramsid=0,residual=False):
    testsize = 10000
    testsize_batch = 500
    network = lcnet.Unet()
    if use_gpu:
        network = nn.DataParallel(network).cuda()
    
    network.load_state_dict(torch.load(path_params+preload_Netmodel[:-4]+"_"+str(paramsid)+".pkl"))

    network.eval()

    valdata = lcnet.Mydataset(n_lc=testsize,data_root=rootval,num_skip=0,residual=residual)
    valset = lcnet.DataLoaderX(valdata, batch_size=testsize_batch,shuffle=True,num_workers=num_process,pin_memory=True)

    network.eval()

    test_list_label = np.array([])
    test_list_output = np.array([])
    with torch.no_grad():
        for j,valdata in enumerate(valset):
            val_inputs, val_labels = valdata
            val_inputs = val_inputs.float()
            if use_gpu:
                val_inputs = val_inputs.cuda()
                val_labels = val_labels.cuda()
            val_outputs = network(val_inputs).detach().cpu().numpy()
            val_labels = val_labels.detach().cpu().numpy()

            for i in range(val_outputs.shape[0]):
                test_list_label = np.append(test_list_label,val_labels[i])
                test_list_output = np.append(test_list_output,val_outputs[i][1])

    print(test_list_label.shape)
    print(test_list_output.shape)

    thres_list = np.linspace(0.999,1,100)
    accuracy_list = []
    for thres_test in thres_list:
        accuracy_list.append(np.mean((test_list_output > thres_test)))

    print(np.mean(test_list_label))

    plt.figure()
    plt.plot(thres_list,accuracy_list)
    plt.plot(thres_list,np.mean(test_list_label)*np.ones(thres_list.shape))
    plt.savefig("accu_thres_500_%d.png"%paramsid)
    plt.close()


def testUnet_KMT2019(posi, network=None,network_ref=None, thres=0.999, thres_ref=0.998):
    print("index: ", posi)
    KMT_args,data_total,data_A,data_C,data_S,_ = loaddata.getKMTdata(posi=posi,cutratio=[-3,3],FWHM_threshold=50,sky_threshold=20000)
    time,mag,err = data_total

    minimize_args, mag_singlemodel = loaddata.doublefitting(time,mag,err,KMT_args)

    new_cutratio = minimize_args[0]/KMT_args[0]
    new_correctratio = (minimize_args[1]-KMT_args[1])/KMT_args[0]

    print("t0: ",minimize_args[1],KMT_args[1])

    KMT_args,data_total,data_A,data_C,data_S,_ = loaddata.getKMTdata(posi=posi,cutratio=[-2*new_cutratio+new_correctratio,2*new_cutratio+new_correctratio],FWHM_threshold=50,sky_threshold=20000)
    time,mag,err = data_total
    print("cut_ratio: ", new_cutratio)
    print("size: ", len(time))

    if len(time) < 500:
        return 

    time_rs,mag_rs,err_rs = lcnet.sample_curve(time,mag,err,length_resample=500)
    mag_single_rs = loaddata.mag_cal(time_rs,*minimize_args)

    # 2 samples of data: with residual and no residual
    data_input_ref = np.array(lcnet.loader_transform(time_rs,mag_rs,err_rs,size_check=500))
    data_input = np.array(lcnet.loader_transform(time_rs,((mag_rs-mag_single_rs)/err_rs)**2,err_rs,size_check=500))

    data_input = torch.from_numpy(data_input).float()
    data_input_ref = torch.from_numpy(data_input_ref).float()
    if use_gpu:
        data_input = data_input.cuda()
        data_input_ref = data_input_ref.cuda()
    
    network.eval()
    network_ref.eval()
    with torch.no_grad():
        data_output = network(data_input).detach().cpu().numpy()
        data_output_ref = network_ref(data_input_ref).detach().cpu().numpy()

    print("Output of network: ", data_output.shape)

    bspre_array = data_output[0][1]
    bspre_array_ref = data_output_ref[0][1]
    
    bspre_sigmoid = lcnet.sigmoid_unettest(bspre_array,center=thres,scale=2.5)
    bspre_sigmoid_ref = lcnet.sigmoid_unettest(bspre_array_ref,center=thres_ref,scale=2.5)

    plt.figure(figsize=(10,16))
    # plt.subplot(211)
    plt.axes([0.1, 0.696, 0.72, 0.273])
    plt.scatter(time_rs, mag_rs, s=10)
    plt.errorbar(time_rs,mag_rs,yerr=err_rs,fmt='o',capsize=2,elinewidth=1,ms=0,alpha=0.7,c="blue")
    plt.errorbar(time_rs[bspre_array>thres],mag_rs[bspre_array>thres],yerr=err_rs[bspre_array>thres],fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,c="red", label="with residual")
    plt.errorbar(time_rs[bspre_array_ref>thres_ref],mag_rs[bspre_array_ref>thres_ref],yerr=err_rs[bspre_array_ref>thres_ref],fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7,c="green", label="without residual")
    plt.plot(time_rs,mag_single_rs,ls="--")
    plt.xlabel("time/HJD")
    plt.ylabel("magnitude")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title("KMT-2019-%04d, threshold=%.3f"%(posi,thres))
    # plt.subplot(212)
    plt.axes([0.1, 0.363, 0.9, 0.273])
    plt.scatter(time_rs, mag_rs, s=10, c=bspre_sigmoid, cmap=plt.cm.Reds, edgecolors='none')
    plt.plot(time_rs,mag_single_rs,ls="--")
    plt.xlabel("time/HJD")
    plt.ylabel("magnitude")
    plt.gca().invert_yaxis()
    plt.title(r"$sigmoid(2.5*\frac{label-%.3f}{1-%.3f})$"%(thres,thres,))
    plt.colorbar()
    plt.axes([0.1, 0.03, 0.9, 0.273])
    plt.scatter(time_rs, mag_rs, s=10, c=bspre_sigmoid_ref, cmap=plt.cm.Reds, edgecolors='none')
    plt.plot(time_rs,mag_single_rs,ls="--")
    plt.xlabel("time/HJD")
    plt.ylabel("magnitude")
    plt.gca().invert_yaxis()
    plt.title(r"$sigmoid(2.5*\frac{label-%.3f}{1-%.3f})$"%(thres_ref,thres_ref,))
    plt.colorbar()
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/realfig/binary/KMT-2019-%04d.pdf"%(posi,))
    plt.close()

def params_loader(paramsid,params_name):
    network = lcnet.Unet()
    if use_gpu:
        network = nn.DataParallel(network).cuda()
    network.load_state_dict(torch.load(path_params+params_name[:-4]+"_"+str(paramsid)+".pkl"))
    return network
    
def get_dchis_executer(dataoris, labels, predicts, size, thres, scaled=False):
    # dataori: [time, lc_withnoi, err, lc_withoutnoi, lc_singlemodel], (size, 5, length)
    # label: (size, length)
    # predict: (size, 2, length)
    dchis_label_list = []
    dchis_predict_list = []
    length_label_list = []
    length_predict_list = []
    for i in range(size):
        time, lc_withnoi, err, lc_withoutnoi, lc_singlemodel = dataoris[i]
        label = labels[i]
        predict = (predicts[i][1] > thres).astype(np.int)

        dchis_label = lcnet.dchis_cal(lc_withnoi,lc_withoutnoi,lc_singlemodel,err,label)
        dchis_predict = lcnet.dchis_cal(lc_withnoi,lc_withoutnoi,lc_singlemodel,err,predict)

        dchis_label_list.append(dchis_label)
        dchis_predict_list.append(dchis_predict)
        length_label_list.append(np.sum(label))
        length_predict_list.append(np.sum(predict))

    dchis_label_list = np.array(dchis_label_list)
    dchis_predict_list = np.array(dchis_predict_list)
    length_label_list = np.array(length_label_list)
    length_predict_list = np.array(length_predict_list)

    dchis_label_list = dchis_label_list[length_predict_list>0]
    dchis_predict_list = dchis_predict_list[length_predict_list>0]
    length_label_list = length_label_list[length_predict_list>0]
    length_predict_list = length_predict_list[length_predict_list>0]
    
    print(np.mean(length_predict_list))
    print(np.sum(length_predict_list<=0))
    print(np.mean(length_label_list))
    print(np.sum(length_label_list<=0))

    if scaled:
        return dchis_predict_list/length_predict_list,dchis_label_list/length_label_list, length_predict_list, length_label_list
    else:
        return dchis_predict_list,dchis_label_list, length_predict_list, length_label_list



    
def scatter_comparer(num_test, num_skip, network_res, network_ref, thres_res, thres_ref, border=(0.5,7.5),scaled=False):
    valdata_res = lcnet.Mydataset(n_lc=num_test,data_root=rootval,residual=True,loader=lcnet.loader_grouptest)
    valset_res = lcnet.DataLoaderX(valdata_res, batch_size=num_test//2,shuffle=False,num_workers=num_process,pin_memory=True)
    valdata_ref = lcnet.Mydataset(n_lc=num_test,data_root=rootval,residual=False,loader=lcnet.loader_grouptest)
    valset_ref = lcnet.DataLoaderX(valdata_ref, batch_size=num_test//2,shuffle=False,num_workers=num_process,pin_memory=True)

    dchis_res_predict_total = np.array([])
    dchis_ref_predict_total = np.array([])

    dchis_res_label_total = np.array([])
    dchis_ref_label_total = np.array([])

    length_res_predict_total = np.array([])
    length_ref_predict_total = np.array([])

    length_res_label_total = np.array([])
    length_ref_label_total = np.array([])

    with torch.no_grad():
        for i, val_res in enumerate(valset_res):
            val_inputs_res, val_labelanddata_res = val_res
            val_labels_res = val_labelanddata_res[:,0,:].long()
            val_dataori_res = val_labelanddata_res[:,1:,:].numpy()
            val_inputs_res = val_inputs_res.float()
            if use_gpu:
                val_inputs_res = val_inputs_res.cuda()
                val_labels_res = val_labels_res.cuda()
            val_outputs_res = network_res(val_inputs_res).detach().cpu().numpy()
            val_inputs_res = val_inputs_res.detach().cpu().numpy()
            val_labels_res = val_labels_res.detach().cpu().numpy()

            dchis_pre_res, dchis_label_res, length_pre_res, length_label_res = get_dchis_executer(dataoris=val_dataori_res, labels=val_labels_res, predicts=val_outputs_res, size=val_labels_res.shape[0], thres=thres_res, scaled=scaled)
            print(dchis_pre_res.shape)
            print(dchis_label_res.shape)
            dchis_res_predict_total = np.append(dchis_res_predict_total,dchis_pre_res)
            dchis_res_label_total = np.append(dchis_res_label_total,dchis_label_res)
            length_res_predict_total = np.append(length_res_predict_total,length_pre_res)
            length_res_label_total = np.append(length_res_label_total,length_label_res)

        for i, val_ref in enumerate(valset_ref):
            val_inputs_ref, val_labelanddata_ref = val_ref
            val_labels_ref = val_labelanddata_ref[:,0,:].long()
            val_dataori_ref = val_labelanddata_ref[:,1:,:].numpy()
            val_inputs_ref = val_inputs_ref.float()
            if use_gpu:
                val_inputs_ref = val_inputs_ref.cuda()
                val_labels_ref = val_labels_ref.cuda()
            val_outputs_ref = network_ref(val_inputs_ref).detach().cpu().numpy()
            val_inputs_ref = val_inputs_ref.detach().cpu().numpy()
            val_labels_ref = val_labels_ref.detach().cpu().numpy()

            dchis_pre_ref, dchis_label_ref, length_pre_ref, length_label_ref = get_dchis_executer(dataoris=val_dataori_ref, labels=val_labels_ref, predicts=val_outputs_ref, size=val_labels_ref.shape[0], thres=thres_ref, scaled=scaled)
            print(dchis_pre_ref.shape)
            print(dchis_label_ref.shape)
            dchis_ref_predict_total = np.append(dchis_ref_predict_total,dchis_pre_ref)
            dchis_ref_label_total = np.append(dchis_ref_label_total,dchis_label_ref)
            length_ref_predict_total = np.append(length_ref_predict_total,length_pre_ref)
            length_ref_label_total = np.append(length_ref_label_total,length_label_ref)
    # xyaxis
    x_label = "$\log_{10}|\Delta \chi^2|$(label)"
    y_label = "$\log_{10}|\Delta \chi^2|$(predict)"
    if scaled:
        x_label = "Scaled " + x_label
        y_label = "Scaled " + y_label
    # border
    line50up_res,line50down_res,linexaxis1_res = dm.get_rate_updown_line(0.5,log10abs(dchis_res_label_total),log10abs(dchis_res_predict_total),np.linspace(0,8,32))
    line75up_res,line75down_res,linexaxis2_res = dm.get_rate_updown_line(0.75,log10abs(dchis_res_label_total),log10abs(dchis_res_predict_total),np.linspace(0,8,32))
    line90up_res,line90down_res,linexaxis3_res = dm.get_rate_updown_line(0.9,log10abs(dchis_res_label_total),log10abs(dchis_res_predict_total),np.linspace(0,8,32))
    
    line50up_ref,line50down_ref,linexaxis1_ref = dm.get_rate_updown_line(0.5,log10abs(dchis_ref_label_total),log10abs(dchis_ref_predict_total),np.linspace(0,8,32))
    line75up_ref,line75down_ref,linexaxis2_ref = dm.get_rate_updown_line(0.75,log10abs(dchis_ref_label_total),log10abs(dchis_ref_predict_total),np.linspace(0,8,32))
    line90up_ref,line90down_ref,linexaxis3_ref = dm.get_rate_updown_line(0.9,log10abs(dchis_ref_label_total),log10abs(dchis_ref_predict_total),np.linspace(0,8,32))


    plt.figure(figsize=(20,20))
    plt.subplot(221)
    plt.scatter(log10abs(dchis_res_label_total),log10abs(dchis_res_predict_total),s=2,c="blue",alpha=0.5)

    plt.fill_between(linexaxis1_res,line50down_res,line50up_res, where=line50down_res<line50up_res, facecolor="orange",alpha=0.5,label="$<50\%$")
    plt.fill_between(linexaxis2_res,line75down_res,line50down_res, where=line75down_res<line50down_res,facecolor="greenyellow", alpha=0.5)
    plt.fill_between(linexaxis2_res,line50up_res,line75up_res, where=line50up_res<line75up_res, facecolor="greenyellow",alpha=0.5,label="$50\sim 75\%$")
    plt.fill_between(linexaxis3_res,line90down_res,line75down_res, where=line90down_res<line75down_res, facecolor="tomato", alpha=0.5)
    plt.fill_between(linexaxis3_res,line75up_res,line90up_res, where=line75up_res<line90up_res, facecolor="tomato", alpha=0.5,label="$75\sim 90\%$")
    plt.plot(border, border,ls="--")
    plt.legend()
    plt.grid()
    plt.axis("scaled")
    plt.xlim(border)
    plt.ylim(border)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Network with residual")

    plt.subplot(222)
    plt.scatter(length_res_label_total, length_res_predict_total,s=2, c="tomato", alpha=0.5)# c=map01(log10abs(dchis_res_label_total)), cmap=plt.cm.Reds)
    plt.plot((0,130), (0,130),ls="--")
    plt.xlabel("length (label)")
    plt.ylabel("length (predict)")
    plt.grid()
    plt.axis("scaled")
    plt.xlim((0,130))
    plt.ylim((0,130))
    plt.title("Length (with residual)")

    plt.subplot(223)
    plt.scatter(log10abs(dchis_ref_label_total),log10abs(dchis_ref_predict_total),s=2,c="blue",alpha=0.5)

    plt.fill_between(linexaxis1_ref,line50down_ref,line50up_ref, where=line50down_ref<line50up_ref, facecolor="orange",alpha=0.5,label="$<50\%$")
    plt.fill_between(linexaxis2_ref,line75down_ref,line50down_ref, where=line75down_ref<line50down_ref,facecolor="greenyellow", alpha=0.5)
    plt.fill_between(linexaxis2_ref,line50up_ref,line75up_ref, where=line50up_ref<line75up_ref, facecolor="greenyellow",alpha=0.5,label="$50\sim 75\%$")
    plt.fill_between(linexaxis3_ref,line90down_ref,line75down_ref, where=line90down_ref<line75down_ref, facecolor="tomato", alpha=0.5)
    plt.fill_between(linexaxis3_ref,line75up_ref,line90up_ref, where=line75up_ref<line90up_ref, facecolor="tomato", alpha=0.5,label="$75\sim 90\%$")
    plt.plot(border, border,ls="--")
    plt.legend()
    plt.grid()
    plt.axis("scaled")
    plt.xlim(border)
    plt.ylim(border)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Network without residual")

    plt.subplot(224)
    plt.scatter(length_ref_label_total, length_ref_predict_total,s=2, c="tomato", alpha=0.5)# c=map01(log10abs(dchis_ref_label_total)), cmap=plt.cm.Reds)
    plt.plot((0,130), (0,130),ls="--")
    plt.xlabel("length(label)")
    plt.ylabel("length(predict)")
    plt.grid()
    plt.axis("scaled")
    plt.xlim((0,130))
    plt.ylim((0,130))
    plt.title("Length (without residual)")

    path_store = "/home/zerui603/MDN_lc_iden/unet/dchis_resVSref.png"
    if scaled:
        path_store = "/home/zerui603/MDN_lc_iden/unet/dchis_resVSref_scaled.png"
    
    plt.savefig(path_store)



if __name__=="__main__":
    # training(paramsid=0)
    
    preload_name_ref = "GRU_unet_500.pkl"
    preload_name_res = "GRU_unet_500_res.pkl"

    network_res = params_loader(60,preload_name_res)
    network_ref = params_loader(80,preload_name_ref)

    scatter_comparer(num_test=10000, num_skip=0, network_res=network_res, network_ref=network_ref, thres_res=0.9, thres_ref=0.9, border=(1.0,7.5), scaled=False)
    scatter_comparer(num_test=10000, num_skip=0, network_res=network_res, network_ref=network_ref, thres_res=0.9, thres_ref=0.9, border=(0.0,5.5), scaled=True)

    '''

    testfig(num_test=100,network=network_res,network_ref=network_ref,thres=0.9999,thres_ref=0.9998,num_skip=1000)

    KMT2019anomaly = np.loadtxt("/home/zerui603/MDN_lc_iden/unet/KMT2019anomaly.txt").astype(np.int64)

    for posi in KMT2019anomaly:
        try:
            testUnet_KMT2019(posi,network_res,network_ref,thres=0.999,thres_ref=0.998)
        except:
            print("ERROR event: ", posi)
            print(traceback.print_exc())

    '''
            
