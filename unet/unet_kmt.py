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

import datamodule.dm as dm
import netmodule.unetforkmt as lcnet

def chis(x1,x2,sig,weight=1):
    return np.sum((np.array(x1)-np.array(x2))**2/np.array(sig)**2*weight)

def chis_array(x1,x2,sig,weight=1):
    return (np.array(x1)-np.array(x2))**2/np.array(sig)**2*weight

        

# reload
reload = 0
preload_Netmodel = "GRU_unet.pkl"
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
learning_rate = 5e-3
stepsize = 10# 7
gamma_0 = 0.8
momentum = 0.5

## path of trainingset and validationset

rootdir = "/scratch/zerui603/KMT_unet/high_ratio/training/"
rootval = "/scratch/zerui603/KMT_unet/high_ratio/val/"

# training

def training(paramsid):
    # Loading datas
    trainingdata = lcnet.Mydataset(n_lc=size_train,data_root=rootdir)
    trainset = lcnet.DataLoaderX(trainingdata, batch_size=batch_size_train,shuffle=True,num_workers=num_process,pin_memory=True)

    valdata = lcnet.Mydataset(n_lc=size_val,data_root=rootval)
    valset = lcnet.DataLoaderX(valdata, batch_size=batch_size_val,shuffle=True,num_workers=num_process,pin_memory=True)

    # initialize model
    network = lcnet.Unet()

    weights = [1.0, 20.0]
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

        if (epoch+1)%2 == 0:
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
        
        plt.savefig("loss_accuracy_Unet_lowratio.png")
        plt.close()

    torch.save(network.state_dict(),path_params+preload_Netmodel)
    np.save("Unet_loss_lowratio.npy",np.array([loss_figure,val_loss_figure]))

def test(paramsid):
    thres = 1-0.9985
    testsize = 100
    testsize_batch = 50
    fig_path = "/home/zerui603/MDN_lc_iden/unet/testfig/"
    network = lcnet.Unet()
    if use_gpu:
        network = nn.DataParallel(network).cuda()
    
    network.load_state_dict(torch.load(path_params+preload_Netmodel[:-4]+"_"+str(paramsid)+".pkl"))

    network.eval()

    valdata = lcnet.Mydataset(n_lc=testsize,data_root=rootval,num_skip=5000,loader=lcnet.loader_fortest)
    valset = lcnet.DataLoaderX(valdata, batch_size=testsize_batch,shuffle=False,num_workers=num_process,pin_memory=True)

    with torch.no_grad():
        for j,valdata in enumerate(valset):
            val_inputs, val_labelanddata = valdata

            val_labels = val_labelanddata[:,0,:].long()

            val_dataori = val_labelanddata[:,1:,:].numpy()

            # print(val_dataori.shape)

            val_inputs = val_inputs.float()
            if use_gpu:
                val_inputs = val_inputs.cuda()
                val_labels = val_labels.cuda()

            
            val_outputs = network(val_inputs).detach().cpu().numpy()
            val_inputs = val_inputs.detach().cpu().numpy()
            val_labels = val_labels.detach().cpu().numpy()

            print(np.mean(val_labels))

            # label_count = []

            for i in range(testsize_batch):
                lc_data = val_inputs[i][0].T
                predict = val_outputs[i][0]
                label  = val_labels[i]


                time,lc_withnoi,err,lc_withoutnoi,lc_singlemodel = val_dataori[i]

                predict_01_array = predict<=thres

                dchis_label_array = chis_array(lc_withnoi,lc_singlemodel,err,label)-chis_array(lc_withnoi,lc_withoutnoi,err,label)
                dchis_predict_array = chis_array(lc_withnoi,lc_singlemodel,err,predict_01_array)-chis_array(lc_withnoi,lc_withoutnoi,err,predict_01_array)
                
                # dchis_label_origin = np.abs(np.sum(dchis_label_array))

                dchis_label = np.abs(np.sum(dchis_label_array))
                dchis_predict = np.abs(np.sum(dchis_predict_array))

                if dchis_label == 0:
                    continue
                

                # label_count.append(np.mean(label))
                s_point = lc_withnoi[predict>thres]
                s_time = time[predict>thres]
                b_point = lc_withnoi[predict<=thres]
                b_time = time[predict<=thres]

                s_point_label = lc_withnoi[label<thres]
                s_time_label = time[label<thres]
                b_point_label = lc_withnoi[label>=thres]
                b_time_label = time[label>=thres]

                mag_max_lim = np.mean(np.sort(lc_withoutnoi)[-25:])
                mag_min_lim = np.mean(np.sort(lc_withoutnoi)[:25])
                mag_max_lim += 0.1*(mag_max_lim-mag_min_lim)
                mag_min_lim -= 0.3*(mag_max_lim-mag_min_lim)

                plt.figure(figsize=(10,5))
                # plt.subplot(311)
                plt.ylim(mag_min_lim,mag_max_lim)
                plt.scatter(s_time,s_point,s=4,alpha=0.5,label = "predict no structure",c="blue")
                plt.scatter(b_time,b_point,s=4,alpha=0.5,label = "predict with structure",c="tomato")
                plt.plot(time,lc_withoutnoi,ls="--",label="binary model",c="green",alpha=0.3)
                plt.plot(time,lc_singlemodel,ls="--",label="single model",c="red",alpha=0.3)
                plt.xlabel("t",fontsize=16)
                plt.ylabel("Mag",fontsize=16)
                plt.legend()
                plt.gca().invert_yaxis()
                plt.savefig("/home/zerui603/MDN_lc_iden/unet/testfig/"+str(np.int(i+j*testsize_batch))+"pre.pdf")
                plt.close()
                
                plt.figure(figsize=(10,5))
                # plt.subplot(312)
                plt.ylim(mag_min_lim,mag_max_lim)
                plt.scatter(s_time_label,s_point_label,s=4,alpha=0.5,label = "label no structure",c="blue")
                plt.scatter(b_time_label,b_point_label,s=4,alpha=0.5,label = "label with structure",c="tomato")
                plt.plot(time,lc_withoutnoi,ls="--",label="binary model",c="green",alpha=0.3)
                plt.plot(time,lc_singlemodel,ls="--",label="single model",c="red",alpha=0.3)
                plt.xlabel("t",fontsize=16)
                plt.ylabel("Mag",fontsize=16)
                plt.legend()
                plt.gca().invert_yaxis()
                # plt.title("%.4f,%.4f"%(np.mean(np.abs(lc_withoutnoi-lc_singlemodel)),np.std(np.abs(lc_withoutnoi-lc_singlemodel))))
                '''
                plt.subplot(313)
                # plt.scatter(s_time_label,s_point_label,s=4,alpha=0.5,label = "label no structure",c="blue")
                plt.scatter(time[label>=0.5],(lc_withoutnoi-lc_singlemodel)[label>=0.5],s=4,alpha=0.5,label = "label with structure",c="red")
                # plt.plot(time,lc_withoutnoi,ls="--",label="binary model",c="green")
                # plt.plot(time,lc_singlemodel,ls="--",label="single model",c="orange")
                plt.xlabel("t",fontsize=16)
                plt.ylabel("Mag",fontsize=16)
                plt.legend()
                plt.gca().invert_yaxis()
                '''
                plt.savefig("/home/zerui603/MDN_lc_iden/unet/testfig/"+str(np.int(i+j*testsize_batch))+"label.pdf")
                plt.close()

def testgif():
    testsize = 40
    testsize_batch = 20
    gif_ori_path = "/scratch/zerui603/KMT_unet/high_ratio/gif_fig/"
    fig_path = "/home/zerui603/MDN_lc_iden/unet/testfig/"

    paramsid_list = (2*np.array(list(range(1,30)))).astype(np.int)#[10,20,30,40,50,60,70,80,90,100]

    valdata = lcnet.Mydataset(n_lc=testsize,data_root=rootval,loader=lcnet.loader_fortest)
    valset = lcnet.DataLoaderX(valdata, batch_size=testsize_batch,shuffle=False,num_workers=num_process,pin_memory=True)

    for index_net,paramsid in enumerate(paramsid_list):
        network = lcnet.Unet()
        if use_gpu:
            network = nn.DataParallel(network).cuda()
        network.load_state_dict(torch.load(path_params+preload_Netmodel[:-4]+"_"+str(paramsid)+".pkl"))
        network.eval()
        with torch.no_grad():
            for j,valdata in enumerate(valset):
                val_inputs, val_labelanddata = valdata

                val_labels = val_labelanddata[:,0,:].long()

                val_dataori = val_labelanddata[:,1:,:].numpy()

                # print(val_dataori.shape)

                val_inputs = val_inputs.float()
                if use_gpu:
                    val_inputs = val_inputs.cuda()
                    val_labels = val_labels.cuda()

                
                val_outputs = network(val_inputs).detach().cpu().numpy()
                val_inputs = val_inputs.detach().cpu().numpy()
                val_labels = val_labels.detach().cpu().numpy()

                print(np.mean(val_labels))

                # label_count = []

                for i in range(testsize_batch):
                    lc_data = val_inputs[i][0].T
                    predict = val_outputs[i][0]
                    label  = val_labels[i]


                    time,lc_withnoi,err,lc_withoutnoi,lc_singlemodel = val_dataori[i]
                    # label_count.append(np.mean(label))
                    s_point = lc_withnoi[predict>0.5]
                    s_time = time[predict>0.5]
                    b_point = lc_withnoi[predict<=0.5]
                    b_time = time[predict<=0.5]

                    s_point_label = lc_withnoi[label<0.5]
                    s_time_label = time[label<0.5]
                    b_point_label = lc_withnoi[label>=0.5]
                    b_time_label = time[label>=0.5]

                    plt.figure(figsize=(12,18))
                    plt.subplot(211)
                    plt.scatter(s_time,s_point,s=4,alpha=0.5,label = "predict no structure",c="blue")
                    plt.scatter(b_time,b_point,s=4,alpha=0.5,label = "predict with structure",c="red")
                    plt.plot(time,lc_withoutnoi,ls="--",label="binary model",c="green")
                    plt.plot(time,lc_singlemodel,ls="--",label="single model",c="orange")
                    plt.xlabel("t",fontsize=16)
                    plt.ylabel("Mag",fontsize=16)
                    plt.legend()
                    plt.gca().invert_yaxis()
                    plt.subplot(212)
                    plt.scatter(s_time_label,s_point_label,s=4,alpha=0.5,label = "label no structure",c="blue")
                    plt.scatter(b_time_label,b_point_label,s=4,alpha=0.5,label = "label with structure",c="red")
                    plt.plot(time,lc_withoutnoi,ls="--",label="binary model",c="green")
                    plt.plot(time,lc_singlemodel,ls="--",label="single model",c="orange")
                    plt.xlabel("t",fontsize=16)
                    plt.ylabel("Mag",fontsize=16)
                    plt.legend()
                    plt.gca().invert_yaxis()
                    plt.suptitle("ID of netparams: %d"%(paramsid,))
                    plt.savefig(gif_ori_path+str(np.int(i+j*testsize_batch))+"_"+str(index_net)+".png")
                    plt.close()

    for index_event in range(testsize):
        img_paths = [gif_ori_path+"%d_%d.png"%(index_event,x) for x in range(len(paramsid_list))]
        gif_images = []
        for path in img_paths:
            gif_images.append(imageio.imread(path))
        imageio.mimsave(fig_path+"%d.gif"%(index_event,),gif_images,fps=2)

def teststatic(paramsid):
    testsize = 10000
    testsize_batch = 500
    fig_path = "/home/zerui603/MDN_lc_iden/unet/testfig/"
    network = lcnet.Unet()
    if use_gpu:
        network = nn.DataParallel(network).cuda()
    
    network.load_state_dict(torch.load(path_params+preload_Netmodel[:-4]+"_"+str(paramsid)+".pkl"))

    network.eval()

    valdata = lcnet.Mydataset(n_lc=testsize,data_root=rootval,num_skip=10000,loader=lcnet.loader_fortest)
    valset = lcnet.DataLoaderX(valdata, batch_size=testsize_batch,shuffle=True,num_workers=num_process,pin_memory=False)

    dchis_pre_list = []
    dchis_label_list = []

    label_list = []
    predicted_list = []

    print("check")

    with torch.no_grad():
        for j,valdata in enumerate(valset):
            # if j > 4:
            #     break
            val_inputs, val_labelanddata = valdata

            val_labels = val_labelanddata[:,0,:].long()

            val_dataori = val_labelanddata[:,1:,:].numpy()

            # print(val_dataori.shape)

            val_inputs = val_inputs.float()
            if use_gpu:
                val_inputs = val_inputs.cuda()
                val_labels = val_labels.cuda()

            
            val_outputs = network(val_inputs).detach().cpu().numpy()
            val_inputs = val_inputs.detach().cpu().numpy()
            val_labels = val_labels.detach().cpu().numpy()

            # label_count = []

            for i in range(testsize_batch):
                lc_data = val_inputs[i][0].T
                predict = val_outputs[i][0]
                label  = val_labels[i]
                
                time,lc_withnoi,err,lc_withoutnoi,lc_singlemodel = val_dataori[i]
                if j < 0:
                    plt.figure()
                    plt.errorbar(time,lc_withnoi,yerr=err,fmt='o',capsize=2,elinewidth=1,ms=1,alpha=0.7)
                    plt.plot(time,lc_withoutnoi,ls="--",label="binary model",c="green")
                    plt.plot(time,lc_singlemodel,ls="--",label="single model",c="orange")
                    plt.xlabel("t",fontsize=16)
                    plt.ylabel("Mag",fontsize=16)
                    plt.gca().invert_yaxis()
                    plt.legend()
                    plt.savefig("/home/zerui603/MDN_lc_iden/unet/testfig/test_"+str(i)+".png")


                # label_count.append(np.mean(label))
                s_point = lc_withnoi[predict>0.5]
                s_time = time[predict>0.5]
                b_point = lc_withnoi[predict<=0.5]
                b_time = time[predict<=0.5]

                b_binarymodel = lc_withoutnoi[predict<=0.5]
                b_singlemodel = lc_singlemodel[predict<=0.5]
                b_err = err[predict<=0.5]

                predict_01 = predict <= 0.5

                if i%10 == 0:
                    predicted_list = np.append(predicted_list,1-predict)
                    label_list = np.append(label_list,label)

                predict_01 = predict_01.astype(np.int)

                delta_chi2 = np.sum((lc_withnoi-lc_singlemodel)**2/err**2-(lc_withnoi-lc_withoutnoi)**2/err**2)

                delta_chi2_pre = np.sum(predict_01*(lc_withnoi-lc_singlemodel)**2/err**2-predict_01*(lc_withnoi-lc_withoutnoi)**2/err**2)

                dchis_pre_list.append(delta_chi2_pre)
                dchis_label_list.append(delta_chi2)


    lgdchis_pre_list = np.log10(np.abs(dchis_pre_list)+1)
    lgdchis_label_list = np.log10(np.abs(dchis_label_list)+1)
    binsize = 0.2
    plt.figure()
    plt.subplot(211)
    # np.int((np.max(lgdchis_label_list)-np.min(lgdchis_label_list))//binsize)
    plt.hist(lgdchis_pre_list,bins=100,label="predicted $\log_{10}|\Delta\chi^2|$",histtype="step")
    plt.hist(lgdchis_label_list,bins=100,label="label $\log_{10}|\Delta\chi^2|$",histtype="step")
    plt.xlabel("$\log_{10}|\Delta\chi^2|$")
    plt.legend()
    plt.subplot(212)
    plt.hist(np.array(dchis_pre_list)/np.array(dchis_label_list),bins=50,range=(0,1))
    plt.xlabel("$\Delta\chi^2(predicted)/\Delta\chi^2(label)$")
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/hist_chis_unet_distribution.png")
    plt.close()

    print(np.sort(predicted_list)[np.int(0.95*len(predicted_list))])
    
    threshold_list = 10**np.linspace(np.log10(0.8),0,100)
    accuracy_list = []
    FPR_list = []
    positive_list = []

    for threshold in threshold_list:
        accuracy_list.append(np.sum((predicted_list > threshold).astype(np.int)==label_list.astype(np.int))/len(predicted_list))
        FPR_list.append(np.sum((predicted_list < threshold).astype(np.int)*label_list.astype(np.int))/len(predicted_list))
        positive_list.append(np.mean(predicted_list > threshold))
    plt.figure(figsize=(12,12))
    plt.subplot(211)
    # plt.plot(threshold_list,accuracy_list)
    # plt.plot(threshold_list,FPR_list)
    plt.plot(threshold_list,np.mean(label_list)*np.ones(threshold_list.shape))
    plt.plot(threshold_list,positive_list)
    plt.xlabel("threshold")
    plt.ylabel("$\log_{10} $num_points")
    plt.subplot(212)
    plt.hist(predicted_list,bins=100)
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/accuracy_threshold.png")
    
    print(threshold_list[np.argmax(accuracy_list)])
    
def teststatic2(paramsid):
    testsize = 10000
    testsize_batch = 500
    fig_path = "/home/zerui603/MDN_lc_iden/unet/testfig/"
    network = lcnet.Unet()
    if use_gpu:
        network = nn.DataParallel(network).cuda()
    
    network.load_state_dict(torch.load(path_params+preload_Netmodel[:-4]+"_"+str(paramsid)+".pkl"))

    network.eval()

    valdata = lcnet.Mydataset(n_lc=testsize,data_root=rootval,num_skip=0,loader=lcnet.loader_fortest)
    valset = lcnet.DataLoaderX(valdata, batch_size=testsize_batch,shuffle=True,num_workers=num_process,pin_memory=False)

    valdata2 = lcnet.Mydataset(n_lc=testsize,data_root=rootval,num_skip=10000,loader=lcnet.loader_fortest)
    valset2 = lcnet.DataLoaderX(valdata2, batch_size=testsize_batch,shuffle=True,num_workers=num_process,pin_memory=False)

    dchis_pre_list = []
    dchis_label_list = []
    label_list = []
    predicted_list = []

    dchis_pre_list_2 = []
    dchis_label_list_2 = []
    label_list_2 = []
    predicted_list_2 = []

    print("check")

    with torch.no_grad():
        for j,valdata in enumerate(valset):
            # if j > 4:
            #     break
            val_inputs, val_labelanddata = valdata

            val_labels = val_labelanddata[:,0,:].long()

            val_dataori = val_labelanddata[:,1:,:].numpy()

            # print(val_dataori.shape)

            val_inputs = val_inputs.float()
            if use_gpu:
                val_inputs = val_inputs.cuda()
                val_labels = val_labels.cuda()

            
            val_outputs = network(val_inputs).detach().cpu().numpy()
            val_inputs = val_inputs.detach().cpu().numpy()
            val_labels = val_labels.detach().cpu().numpy()

            # label_count = []

            for i in range(testsize_batch):
                lc_data = val_inputs[i][0].T
                predict = val_outputs[i][0]
                label  = val_labels[i]

                predict_01 = predict <= 0.5

                if i%5 == 0:
                    predicted_list = np.append(predicted_list,1-predict)
                    label_list = np.append(label_list,label)

        for j,valdata in enumerate(valset2):
            # if j > 4:
            #     break
            val_inputs, val_labelanddata = valdata

            val_labels = val_labelanddata[:,0,:].long()

            val_dataori = val_labelanddata[:,1:,:].numpy()

            # print(val_dataori.shape)

            val_inputs = val_inputs.float()
            if use_gpu:
                val_inputs = val_inputs.cuda()
                val_labels = val_labels.cuda()

            
            val_outputs = network(val_inputs).detach().cpu().numpy()
            val_inputs = val_inputs.detach().cpu().numpy()
            val_labels = val_labels.detach().cpu().numpy()

            # label_count = []

            for i in range(testsize_batch):
                lc_data = val_inputs[i][0].T
                predict = val_outputs[i][0]
                label  = val_labels[i]

                if i%5 == 0:
                    predicted_list_2 = np.append(predicted_list_2,1-predict)
                    label_list_2 = np.append(label_list_2,label)


    
    threshold_list = 10**np.linspace(np.log10(0.9999),np.log10(1),100)
    accuracy_list = []
    FPR_list = []
    positive_list = []
    positive_list2 = []
    

    for threshold in threshold_list:
        accuracy_list.append(np.sum((predicted_list > threshold).astype(np.int)==label_list.astype(np.int))/len(predicted_list))
        FPR_list.append(np.sum((predicted_list < threshold).astype(np.int)*label_list.astype(np.int))/len(predicted_list))
        positive_list.append(np.mean(predicted_list > threshold))
        positive_list2.append(np.mean(predicted_list_2 > threshold))
    plt.figure(figsize=(12,12))
    plt.subplot(211)
    # plt.plot(threshold_list,accuracy_list)
    # plt.plot(threshold_list,FPR_list)
    plt.plot(threshold_list,np.log10(np.mean(label_list))*np.ones(threshold_list.shape))
    plt.plot(threshold_list,np.log10(positive_list),label="predicted positive in binary events")
    # plt.plot(threshold_list,np.log10(np.mean(label_list_2))*np.ones(threshold_list.shape))
    plt.plot(threshold_list,np.log10(positive_list2),label="predicted positive in single events")
    plt.xlabel("threshold")
    plt.ylabel("accuracy")
    plt.legend()
    plt.subplot(212)
    plt.hist(predicted_list,bins=250)
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/accuracy_threshold.pdf")
    
    print(threshold_list[np.argmax(accuracy_list)])

def test_distribution(paramsid,droppoint=0):
    thres = 1-0.9985
    testsize = 10000
    testsize_batch = 1000
    pre_0 = 0
    label_0 = 0
    preandlabel_0 = 0
    fig_path = "/home/zerui603/MDN_lc_iden/unet/testfig/"
    network = lcnet.Unet()
    if use_gpu:
        network = nn.DataParallel(network).cuda()
    network.load_state_dict(torch.load(path_params+preload_Netmodel[:-4]+"_"+str(paramsid)+".pkl"))
    network.eval()

    valdata = lcnet.Mydataset(n_lc=testsize,data_root=rootval,num_skip=0,loader=lcnet.loader_fortest)
    valset = lcnet.DataLoaderX(valdata, batch_size=testsize_batch,shuffle=False,num_workers=num_process,pin_memory=True)

    dchis_label_list = []
    dchis_label_origin_list = []
    dchis_predict_list = []
    dchis_origin_list = []

    rate_cover_list = []
    rate_wide_list = []

    length_label = []
    length_predict = []

    count_bias = 0
    count_plot = 0

    with torch.no_grad():
        for j,valdata in enumerate(valset):
            val_inputs, val_labelanddata = valdata
            val_labels = val_labelanddata[:,0,:].long()
            val_dataori = val_labelanddata[:,1:,:].numpy()
            val_inputs = val_inputs.float()
            if use_gpu:
                val_inputs = val_inputs.cuda()
                val_labels = val_labels.cuda()
            val_outputs = network(val_inputs).detach().cpu().numpy()
            val_inputs = val_inputs.detach().cpu().numpy()
            val_labels = val_labels.detach().cpu().numpy()

            for i in range(testsize_batch):
                lc_data = val_inputs[i][0].T
                predict = val_outputs[i][0]
                label  = val_labels[i]

                time,lc_withnoi,err,lc_withoutnoi,lc_singlemodel = val_dataori[i]

                predict_01_array = predict<=thres

                dchis_label_array = chis_array(lc_withnoi,lc_singlemodel,err,label)-chis_array(lc_withnoi,lc_withoutnoi,err,label)
                dchis_predict_array = chis_array(lc_withnoi,lc_singlemodel,err,predict_01_array)-chis_array(lc_withnoi,lc_withoutnoi,err,predict_01_array)
                
                dchis_label_origin = np.abs(np.sum(dchis_label_array))
                
                droprate = 0.01*droppoint
                # dchis_label_array = np.sort(dchis_label_array)[:-droppoint]# [:np.int((1-droprate)*len(dchis_label_array))]
                # dchis_predict_array = np.sort(dchis_predict_array)[:np.int((1-droprate)*len(dchis_predict_array))]

                dchis_label = np.abs(np.sum(dchis_label_array))
                dchis_predict = np.abs(np.sum(dchis_predict_array))

                if (dchis_predict <= 1)&(dchis_label<=1):
                    preandlabel_0 += 1
                    continue
                if (dchis_predict <= 1)|(dchis_label<=1):
                    if dchis_predict <= 1:
                        pre_0 += 1
                    if dchis_label<=1:
                        label_0 += 1
                    continue

                # testfig
                '''
                if np.log10(dchis_predict) < np.log10(dchis_label)-3: 
                    if count_bias%1 == 0:
                        
                        s_point = lc_withnoi[predict>thres]
                        s_time = time[predict>thres]
                        b_point = lc_withnoi[predict<=thres]
                        b_time = time[predict<=thres]

                        s_point_label = lc_withnoi[label<thres]
                        s_time_label = time[label<thres]
                        b_point_label = lc_withnoi[label>=thres]
                        b_time_label = time[label>=thres]

                        mag_max_lim = np.mean(np.sort(lc_withoutnoi)[-25:])
                        mag_min_lim = np.mean(np.sort(lc_withoutnoi)[:25])
                        mag_max_lim += 0.1*(mag_max_lim-mag_min_lim)
                        mag_min_lim -= 0.1*(mag_max_lim-mag_min_lim)

                        plt.figure(figsize=(10,13))
                        plt.subplot(211)
                        plt.ylim(mag_min_lim,mag_max_lim)
                        plt.scatter(s_time,s_point,s=4,alpha=0.5,label = "predict no structure",c="blue")
                        plt.scatter(b_time,b_point,s=4,alpha=0.5,label = "predict with structure",c="red")
                        plt.plot(time,lc_withoutnoi,ls="--",label="binary model",c="green")
                        plt.plot(time,lc_singlemodel,ls="--",label="single model",c="orange")
                        plt.xlabel("t",fontsize=16)
                        plt.ylabel("Mag",fontsize=16)
                        plt.legend()
                        plt.gca().invert_yaxis()
                        plt.subplot(212)
                        plt.ylim(mag_min_lim,mag_max_lim)
                        plt.scatter(s_time_label,s_point_label,s=4,alpha=0.5,label = "label no structure",c="blue")
                        plt.scatter(b_time_label,b_point_label,s=4,alpha=0.5,label = "label with structure",c="red")
                        plt.plot(time,lc_withoutnoi,ls="--",label="binary model",c="green")
                        plt.plot(time,lc_singlemodel,ls="--",label="single model",c="orange")
                        plt.xlabel("t",fontsize=16)
                        plt.ylabel("Mag",fontsize=16)
                        plt.legend()
                        plt.gca().invert_yaxis()
                        plt.suptitle("predict $log_{10}\Delta \chi^2$ %.4f, label $log_{10}\Delta \chi^2$%.4f"%(np.log10(dchis_predict),np.log10(dchis_label)))
                        plt.savefig("/home/zerui603/MDN_lc_iden/unet/testfig/"+str(np.int(count_bias//1))+"_biaslow.pdf")
                        plt.close()
                    
                    count_bias += 1
                '''
                dchis_label_list.append(dchis_label)
                dchis_predict_list.append(dchis_predict)
                dchis_label_origin_list.append(dchis_label_origin)
                dchis_origin_list.append(np.abs(chis(lc_withnoi,lc_singlemodel,err)-chis(lc_withnoi,lc_withoutnoi,err)))
                rate_cover_list.append(np.sum(predict_01_array*label)/np.sum(label))
                rate_wide_list.append(np.sum(predict_01_array*label)/np.sum(predict_01_array))

                length_label.append(np.sum(label))
                length_predict.append(np.sum(predict_01_array))
    
    plt.figure(figsize=(12,12))
    plt.scatter(np.log10(dchis_label_origin_list),np.log10(dchis_label_list),s=2,zorder=3,alpha=0.5)
    plt.plot(np.linspace(0,8,10),np.linspace(0,8,10),ls="--",c="red")
    plt.xlim((0.5,7.5))
    plt.ylim((0.5,7.5))
    plt.axis("scaled")
    plt.xlabel("$\log_{10}|\Delta \chi^2|$(label)")
    plt.ylabel("$\log_{10}|\Delta \chi^2|$(label,drop %d points)"%(droppoint,))
    plt.legend()
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/drop_compare_%dlow.pdf"%(droppoint,))
    plt.close()

    print(preandlabel_0,pre_0,label_0,len(dchis_label_list))

    # draw origin line

    line50up,line50down,linexaxis1 = dm.get_rate_updown_line(0.5,np.log10(dchis_label_list),np.log10(dchis_predict_list),np.linspace(0,8,32))
    line75up,line75down,linexaxis2 = dm.get_rate_updown_line(0.75,np.log10(dchis_label_list),np.log10(dchis_predict_list),np.linspace(0,8,32))
    line90up,line90down,linexaxis3 = dm.get_rate_updown_line(0.9,np.log10(dchis_label_list),np.log10(dchis_predict_list),np.linspace(0,8,32))
    print(linexaxis1.shape,linexaxis2.shape,linexaxis3.shape)
    plt.figure(figsize=(8,8))
    # plt.subplot(211)
    # plt.title("$\log_{10}|\Delta \chi^2|$")
    plt.scatter(np.log10(dchis_label_list),np.log10(dchis_predict_list),s=1,zorder=3,alpha=0.25)
    plt.plot(np.linspace(0,8,10),np.linspace(0,8,10),ls="--",c="red")
    plt.fill_between(linexaxis1,line50down,line50up, where=line50down<line50up, facecolor="orange",alpha=0.5,label="$<50\%$")
    plt.fill_between(linexaxis2,line75down,line50down, where=line75down<line50down,facecolor="greenyellow", alpha=0.5)
    plt.fill_between(linexaxis2,line50up,line75up, where=line50up<line75up, facecolor="greenyellow",alpha=0.5,label="$50\sim 75\%$")
    plt.fill_between(linexaxis3,line90down,line75down, where=line90down<line75down, facecolor="tomato", alpha=0.5)
    plt.fill_between(linexaxis3,line75up,line90up, where=line75up<line90up, facecolor="tomato", alpha=0.5,label="$75\sim 90\%$")
    plt.axis("scaled")
    plt.xlim((0,7.5))
    plt.ylim((0,7.5))
    plt.xlabel("$\log_{10}|\Delta \chi^2|$(label)")
    plt.ylabel("$\log_{10}|\Delta \chi^2|$(predicted)")
    plt.legend()
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/dchis2D_%d.pdf"%(droppoint,))
    plt.close()
    

    # plt.subplot(212)
    plt.figure(figsize=(8,8))
    # plt.title("$\log_{10}|\Delta \chi^2|/length$")
    length_label = np.array(length_label)
    length_predict = np.array(length_predict)

    length_label = (np.abs(length_label)-0.5)+0.5
    length_predict = (np.abs(length_predict)-0.5)+0.5

    line50up,line50down,linexaxis1 = dm.get_rate_updown_line(0.5,np.log10(dchis_label_list/length_label),np.log10(dchis_predict_list/length_predict),np.linspace(-1,5.5,32))
    line75up,line75down,linexaxis2 = dm.get_rate_updown_line(0.75,np.log10(dchis_label_list/length_label),np.log10(dchis_predict_list/length_predict),np.linspace(-1,5.5,32))
    line90up,line90down,linexaxis3 = dm.get_rate_updown_line(0.9,np.log10(dchis_label_list/length_label),np.log10(dchis_predict_list/length_predict),np.linspace(-1,5.5,32))
    print(linexaxis1.shape,linexaxis2.shape,linexaxis3.shape)
    
    plt.scatter(np.log10(dchis_label_list/length_label),np.log10(dchis_predict_list/length_predict),s=1,zorder=3,alpha=0.25)
    plt.plot(np.linspace(-1,6,10),np.linspace(-1,6,10),ls="--",c="red")
    plt.fill_between(linexaxis1,line50down,line50up, where=line50down<line50up, facecolor="orange",alpha=0.5,label="$<50\%$")
    plt.fill_between(linexaxis2,line75down,line50down, where=line75down<line50down,facecolor="greenyellow", alpha=0.5,label="$50\sim 75\%$")
    plt.fill_between(linexaxis2,line50up,line75up, where=line50up<line75up, facecolor="greenyellow",alpha=0.5)
    plt.fill_between(linexaxis3,line90down,line75down, where=line90down<line75down, facecolor="tomato", alpha=0.5,label="$75\sim 90\%$")
    plt.fill_between(linexaxis3,line75up,line90up, where=line75up<line90up, facecolor="tomato", alpha=0.5)
    plt.axis("scaled")
    plt.xlim(-1,6)
    plt.ylim(-1,6)
    plt.xlabel("$\log_{10}|\Delta \chi^2|/length$(label)")
    plt.ylabel("$\log_{10}|\Delta \chi^2|/length$(predicted)")
    plt.legend()
    # plt.suptitle("Drop %d points"%(droppoint,))
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/dchis2Ds_%d.pdf"%(droppoint,))
    plt.close()
                
    '''
    # select
    print(len(dchis_predict_list))
    lgdchis_label_origin_list = np.log10(dchis_label_origin_list)
    lgdchis_label_list = np.log10(dchis_label_list)

    compare_list = np.abs(lgdchis_label_origin_list-lgdchis_label_list)
    compare_index = np.argwhere(compare_list<0.2).T[0]

    dchis_label_list = np.array(dchis_label_list)[compare_index]
    dchis_predict_list = np.array(dchis_predict_list)[compare_index]
    length_label = np.array(length_label)[compare_index]
    length_predict = np.array(length_predict)[compare_index]
    print(dchis_predict_list.shape)

    # draw line

    line50up,line50down,linexaxis1 = dm.get_rate_updown_line(0.6526,np.log10(dchis_label_list),np.log10(dchis_predict_list),np.linspace(0,8,32))
    line75up,line75down,linexaxis2 = dm.get_rate_updown_line(0.9544,np.log10(dchis_label_list),np.log10(dchis_predict_list),np.linspace(0,8,32))
    line90up,line90down,linexaxis3 = dm.get_rate_updown_line(0.9974,np.log10(dchis_label_list),np.log10(dchis_predict_list),np.linspace(0,8,32))
    print(linexaxis1.shape,linexaxis2.shape,linexaxis3.shape)
    plt.figure(figsize=(12,24))
    plt.subplot(211)
    plt.title("$\log_{10}|\Delta \chi^2|$")
    plt.scatter(np.log10(dchis_label_list),np.log10(dchis_predict_list),s=2,zorder=3,alpha=0.5)
    plt.plot(np.linspace(0,8,10),np.linspace(0,8,10),ls="--",c="red")
    plt.fill_between(linexaxis1,line50down,line50up, where=line50down<line50up, facecolor="orange",alpha=0.5,label="$<1\sigma$")
    plt.fill_between(linexaxis2,line75down,line50down, where=line75down<line50down,facecolor="greenyellow", alpha=0.5,label="$1\sigma\sim 2\sigma$")
    plt.fill_between(linexaxis2,line50up,line75up, where=line50up<line75up, facecolor="greenyellow",alpha=0.5,label="$1\sigma\sim 2\sigma$")
    plt.fill_between(linexaxis3,line90down,line75down, where=line90down<line75down, facecolor="tomato", alpha=0.5,label="$2\sigma\sim 3\sigma$")
    plt.fill_between(linexaxis3,line75up,line90up, where=line75up<line90up, facecolor="tomato", alpha=0.5,label="$2\sigma\sim 3\sigma$")
    plt.xlim((0,8))
    plt.ylim((0,8))
    plt.axis("scaled")
    plt.xlabel("$\log_{10}|\Delta \chi^2|$(label)")
    plt.ylabel("$\log_{10}|\Delta \chi^2|$(predicted)")
    plt.legend()
    plt.subplot(212)

    plt.title("$\log_{10}|\Delta \chi^2|/length$")
    length_label = np.array(length_label)
    length_predict = np.array(length_predict)

    length_label = (np.abs(length_label)-0.5)+0.5
    length_predict = (np.abs(length_predict)-0.5)+0.5

    line50up,line50down,linexaxis1 = dm.get_rate_updown_line(0.6526,np.log10(dchis_label_list/length_label),np.log10(dchis_predict_list/length_predict),np.linspace(-1,5.5,32))
    line75up,line75down,linexaxis2 = dm.get_rate_updown_line(0.9544,np.log10(dchis_label_list/length_label),np.log10(dchis_predict_list/length_predict),np.linspace(-1,5.5,32))
    line90up,line90down,linexaxis3 = dm.get_rate_updown_line(0.9974,np.log10(dchis_label_list/length_label),np.log10(dchis_predict_list/length_predict),np.linspace(-1,5.5,32))
    print(linexaxis1.shape,linexaxis2.shape,linexaxis3.shape)
    
    plt.scatter(np.log10(dchis_label_list/length_label),np.log10(dchis_predict_list/length_predict),s=2,zorder=3,alpha=0.5)
    plt.plot(np.linspace(-1,5.5,10),np.linspace(-1,5.5,10),ls="--",c="red")
    plt.fill_between(linexaxis1,line50down,line50up, where=line50down<line50up, facecolor="orange",alpha=0.5,label="$<1\sigma$")
    plt.fill_between(linexaxis2,line75down,line50down, where=line75down<line50down,facecolor="greenyellow", alpha=0.5,label="$1\sigma\sim 2\sigma$")
    plt.fill_between(linexaxis2,line50up,line75up, where=line50up<line75up, facecolor="greenyellow",alpha=0.5,label="$1\sigma\sim 2\sigma$")
    plt.fill_between(linexaxis3,line90down,line75down, where=line90down<line75down, facecolor="tomato", alpha=0.5,label="$2\sigma\sim 3\sigma$")
    plt.fill_between(linexaxis3,line75up,line90up, where=line75up<line90up, facecolor="tomato", alpha=0.5,label="$2\sigma\sim 3\sigma$")
    plt.xlim((-1,5.5))
    plt.ylim((-1,5.5))
    plt.axis("scaled")
    plt.xlabel("$\log_{10}|\Delta \chi^2|/length$(label)")
    plt.ylabel("$\log_{10}|\Delta \chi^2|/length$(predicted)")
    plt.legend()
    plt.suptitle("Drop %d points"%(droppoint,))
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/dchis2D_select_%d.pdf"%(droppoint,))
    plt.close()
    '''
                


if __name__=="__main__":
    # training(paramsid=0)
    # for i in range(1,9):
    test(paramsid=40)
    # teststatic2(paramsid=40)
    # test_distribution(paramsid=90,droppoint=0)
    # teststatic2(paramsid=90)
    # testgif()
# for i in {0..7};do                        
#     CUDA_VISIBLE_DEVICES=0,1,2,6 python /home/zerui603/MDN_lc_iden/iden_1D/drawhistline.py $i                         
# done