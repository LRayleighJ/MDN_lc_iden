import os
import multiprocessing as mp
import datetime
import numpy as np
import random
import matplotlib.pyplot as plt

rootdir = "/scratch/zerui603/KMT_simu_lowratio/qseries/00to05/"
tempdir = "/scratch/zerui603/KMT_simu_lowratio/qseries/00to05/"
targetdir = "/scratch/zerui603/KMT_simu_lowratio/qseries/00to05/"
file_list = os.listdir(rootdir)

num_files = len(file_list)



def checkdata(index):
    
    data = list(np.load(rootdir+str(index)+".npy", allow_pickle=True))

    labels = data[0]

    bslabel = labels[-1]
    chi_s = labels[-2]
    if bslabel != 0:
        print(index,bslabel,chi_s)

    # os.system("mv "+rootdir+file_list[index]+" "+targetdir)
    

def correctdata(index):
    data = list(np.load(rootdir+str(int(index))+".npy", allow_pickle=True))

    labels = data[0]

    labels[-1] = 0

    data[0] = labels

    np.save(tempdir+str(int(index))+".npy",data,allow_pickle=True)

    command_move = "mv "+tempdir+str(int(index))+".npy"+" "+rootdir+str(int(index))+".npy"
    
    os.system(command_move)

    del data

clean_list = os.listdir(tempdir)
num_clean = len(clean_list)

def clean_files(index):
    os.system("rm -f "+tempdir+clean_list[index])

def chi_square(single,binary,sigma):
    return np.sum(np.power((single-binary)/sigma,2))

def simplifydata(index):
    data = list(np.load(rootdir+str(index)+".npy", allow_pickle=True))
    labels = data[0]
    
    # [times, dtimes, lc_noi, sigma, lc_nonoi, args_minimize, lc_fit_minimize, chi_array]
    times = np.array(data[1])
    lc_withoutnoi = np.array(data[5])
    lc_withnoi = np.array(data[3])
    sig = np.array(data[4])
    chi_s_model = chi_square(lc_withnoi,lc_withoutnoi,sig)

    labels[-2] = labels[-2]-chi_s_model
    lc_res = data[-1]

    data_new = [labels,list(times),list(lc_res),list(sig)]
    np.save(tempdir+str(index)+".npy",data_new,allow_pickle=True)

if __name__=="__main__":
    starttime = datetime.datetime.now()
    print(starttime)
    with mp.Pool(20) as p:
        p.map(checkdata, range(500000,1000000))
    endtime = datetime.datetime.now()
    print(endtime)
    '''
    starttime = datetime.datetime.now()
    with mp.Pool(20) as p:
        p.map(simplifydata, range(1500000))
    endtime = datetime.datetime.now()
    '''

    '''
    args_list = []

    for index in range(1500000):
        data = list(np.load(rootdir+file_list[index], allow_pickle=True))
        labels = data[0]
        
        # [times, dtimes, lc_noi, sigma, lc_nonoi, args_minimize, lc_fit_minimize, chi_array]

        lc_withoutnoi = np.array(data[5])
        lc_withnoi = np.array(data[3])
        sig = np.array(data[4])
        chi_s_model = chi_square(lc_withnoi,lc_withoutnoi,sig)

        labels[-2] = labels[-2]-chi_s_model


        args_list.append(list(np.append(index,labels)))

        del data

    print(len(args_list))

    np.save("args_list_00to05.npy",args_list)

    $$$$$$$
    
    args_list = np.load("/home/zerui603/MDN_lc/datagenerate/args_list_00to05.npy")
    print(np.max(args_list.T[-2]))
    delta_chi_s = args_list.T[-2]

    binary_chi_s = delta_chi_s[delta_chi_s > -1]
    label_bs = args_list.T[-1]

    binary_label_bs = label_bs[delta_chi_s > -1]

    print(binary_chi_s.shape)
    print(np.sum(binary_label_bs))

    plt.figure()
    plt.hist(args_list.T[-2]/np.abs(args_list.T[-2])*np.log10(np.abs(args_list.T[-2])),bins=1000)
    plt.savefig("00to05chis.png")
    plt.close()
    '''
    """
    gen_file_list = os.listdir("/scratch/zerui603/KMT_simu_lowratio/qseries/tempdata/")

    data_mash = np.ones(len(gen_file_list))

    print("number of files:",len(gen_file_list))

    for filename in gen_file_list:
        index = int(filename[:-4])
        try:
            data_mash[index] = 0
        except:
            print(index,"file:",filename)
            continue
    print(np.sum(data_mash))
    """
# 159000+146000

	
