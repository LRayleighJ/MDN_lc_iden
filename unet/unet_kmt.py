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

import netmodule.unetforkmt as lcnet

def chis(x1,x2,sig,weight=1):
    return np.sum((np.array(x1)-np.array(x2))**2/np.array(sig)**2*weight)

        

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
n_epochs = 150
learning_rate = 8e-4
stepsize = 10# 7
gamma_0 = 0.7
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
        
        plt.savefig("loss_accuracy_Unet.png")
        plt.close()

    torch.save(network.state_dict(),path_params+preload_Netmodel)
    np.save("Unet_loss.npy",np.array([loss_figure,val_loss_figure]))

def test(paramsid):
    thres = 1-0.999
    testsize = 100
    testsize_batch = 50
    fig_path = "/home/zerui603/MDN_lc_iden/unet/testfig/"
    network = lcnet.Unet()
    if use_gpu:
        network = nn.DataParallel(network).cuda()
    
    network.load_state_dict(torch.load(path_params+preload_Netmodel[:-4]+"_"+str(paramsid)+".pkl"))

    network.eval()

    valdata = lcnet.Mydataset(n_lc=testsize,data_root=rootval,num_skip=12000,loader=lcnet.loader_fortest)
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
                mag_min_lim -= 0.1*(mag_max_lim-mag_min_lim)

                plt.figure(figsize=(12,18))
                plt.subplot(311)
                plt.ylim(mag_min_lim,mag_max_lim)
                plt.scatter(s_time,s_point,s=4,alpha=0.5,label = "predict no structure",c="blue")
                plt.scatter(b_time,b_point,s=4,alpha=0.5,label = "predict with structure",c="red")
                plt.plot(time,lc_withoutnoi,ls="--",label="binary model",c="green")
                plt.plot(time,lc_singlemodel,ls="--",label="single model",c="orange")
                plt.xlabel("t",fontsize=16)
                plt.ylabel("Mag",fontsize=16)
                plt.legend()
                plt.gca().invert_yaxis()
                plt.subplot(312)
                plt.ylim(mag_min_lim,mag_max_lim)
                plt.scatter(s_time_label,s_point_label,s=4,alpha=0.5,label = "label no structure",c="blue")
                plt.scatter(b_time_label,b_point_label,s=4,alpha=0.5,label = "label with structure",c="red")
                plt.plot(time,lc_withoutnoi,ls="--",label="binary model",c="green")
                plt.plot(time,lc_singlemodel,ls="--",label="single model",c="orange")
                plt.xlabel("t",fontsize=16)
                plt.ylabel("Mag",fontsize=16)
                plt.legend()
                plt.gca().invert_yaxis()
                plt.title("%.4f,%.4f"%(np.mean(np.abs(lc_withoutnoi-lc_singlemodel)),np.std(np.abs(lc_withoutnoi-lc_singlemodel))))
                plt.subplot(313)
                # plt.scatter(s_time_label,s_point_label,s=4,alpha=0.5,label = "label no structure",c="blue")
                plt.scatter(time[label>=0.5],(lc_withoutnoi-lc_singlemodel)[label>=0.5],s=4,alpha=0.5,label = "label with structure",c="red")
                # plt.plot(time,lc_withoutnoi,ls="--",label="binary model",c="green")
                # plt.plot(time,lc_singlemodel,ls="--",label="single model",c="orange")
                plt.xlabel("t",fontsize=16)
                plt.ylabel("Mag",fontsize=16)
                plt.legend()
                plt.gca().invert_yaxis()
                plt.savefig("/home/zerui603/MDN_lc_iden/unet/testfig/"+str(np.int(i+j*testsize_batch))+".png")
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
    
    threshold_list = 10**np.linspace(np.log10(0.99),0,100)
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
    plt.ylabel("accuracy")
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
                    predicted_list_2 = np.append(predicted_list_2,1-predict)
                    label_list_2 = np.append(label_list_2,label)

                predict_01 = predict_01.astype(np.int)

                delta_chi2 = np.sum((lc_withnoi-lc_singlemodel)**2/err**2-(lc_withnoi-lc_withoutnoi)**2/err**2)

                delta_chi2_pre = np.sum(predict_01*(lc_withnoi-lc_singlemodel)**2/err**2-predict_01*(lc_withnoi-lc_withoutnoi)**2/err**2)

                dchis_pre_list_2.append(delta_chi2_pre)
                dchis_label_list_2.append(delta_chi2)


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
    
    threshold_list = 10**np.linspace(np.log10(0.99),0,100)
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
    plt.plot(threshold_list,np.log10(positive_list))
    plt.plot(threshold_list,np.log10(np.mean(label_list_2))*np.ones(threshold_list.shape))
    plt.plot(threshold_list,np.log10(positive_list2))
    plt.xlabel("threshold")
    plt.ylabel("accuracy")
    plt.subplot(212)
    plt.hist(predicted_list,bins=100)
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/accuracy_threshold.png")
    
    print(threshold_list[np.argmax(accuracy_list)])

def test_distribution(paramsid):
    thres = 1-0.999
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
    dchis_predict_list = []
    dchis_origin_list = []

    rate_cover_list = []
    rate_wide_list = []

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

                dchis_label = np.abs(chis(lc_withnoi,lc_singlemodel,err,label)-chis(lc_withnoi,lc_withoutnoi,err,label))
                dchis_predict = np.abs(chis(lc_withnoi,lc_singlemodel,err,predict_01_array)-chis(lc_withnoi,lc_withoutnoi,err,predict_01_array))

                if (dchis_predict <= 1)&(dchis_label<=1):
                    preandlabel_0 += 1
                    continue
                if (dchis_predict <= 1)|(dchis_label<=1):
                    if dchis_predict <= 1:
                        pre_0 += 1
                    if dchis_label<=1:
                        label_0 += 1
                    continue

                dchis_label_list.append(dchis_label)
                dchis_predict_list.append(dchis_predict)
                dchis_origin_list.append(np.abs(chis(lc_withnoi,lc_singlemodel,err)-chis(lc_withnoi,lc_withoutnoi,err)))
                rate_cover_list.append(np.sum(predict_01_array*label)/np.sum(label))
                rate_wide_list.append(np.sum(predict_01_array*label)/np.sum(predict_01_array))

    plt.figure(figsize=(10,20))
    plt.subplot(311)
    plt.scatter(np.log10(dchis_label_list),np.log10(dchis_predict_list),s=2,alpha=0.5)
    plt.plot(np.linspace(0,8,10),np.linspace(0,8,10),ls="--",c="red")
    plt.xlim((0,8))
    plt.ylim((0,8))
    plt.axis("scaled")
    plt.xlabel("$\log_{10}|\Delta \chi^2|$(label)")
    plt.ylabel("$\log_{10}|\Delta \chi^2|$(predicted)")
    plt.subplot(312)
    plt.scatter(np.log10(dchis_label_list),rate_cover_list,label="TP/T",s=2,alpha=0.5)
    plt.scatter(np.log10(dchis_label_list),rate_wide_list,label="TP/P",s=2,alpha=0.5)
    plt.legend()
    plt.xlabel("$\log_{10}|\Delta \chi^2|$")
    plt.ylabel("$rate$")
    plt.subplot(313)
    print(np.mean(dchis_origin_list))
    print(np.mean(dchis_predict_list))
    plt.hist(np.log10(dchis_origin_list),bins=40,range=(0,8),label="origin binary",histtype="step")
    plt.hist(np.log10(dchis_label_list),bins=40,range=(0,8),label="label binary $\log_{10}|\Delta \chi^2|$",histtype="step")
    plt.hist(np.log10(dchis_predict_list),bins=40,range=(0,8),label="predicted $\log_{10}|\Delta \chi^2|$",histtype="step")
    plt.xlabel("$\log_{10}|\Delta \chi^2|$")
    plt.legend()
    plt.savefig("/home/zerui603/MDN_lc_iden/unet/dchis2D.pdf")
    plt.close()
    print(len(dchis_label_list))
    print(pre_0,label_0,preandlabel_0)
                


if __name__=="__main__":
    test_distribution(paramsid=50)
    # teststatic2(paramsid=50)
    # testgif()
# for i in {0..7};do                        
#     CUDA_VISIBLE_DEVICES=0,1,2,6 python /home/zerui603/MDN_lc_iden/iden_1D/drawhistline.py $i                         
# done