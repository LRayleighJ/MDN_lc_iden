import os
import multiprocessing as mp
import datetime
import numpy as np
import random
import warnings

warnings.filterwarnings('error')

rootdir = "/scratch/zerui603/KMTsimudata/sin_training_rename/"
targetdir = "/scratch/zerui603/KMTiden_1d/t_single/"

sindir = "/scratch/zerui603/KMTiden_1d/t_single/"
bindir = "/scratch/zerui603/KMTiden_1d/t_binary/"
targetdir_mix = "/scratch/zerui603/KMTiden_1d/training/"
targetdir_mix_val = "/scratch/zerui603/KMTiden_1d/val/"
bin_num = 750000
sin_num = 750000
bin_num_val = 30000
sin_num_val = 30000

filename_list_sin = os.listdir(sindir)
filename_list_bin = os.listdir(bindir)
filename_list = os.listdir(rootdir)
# random.shuffle(filename_list_sin)
# random.shuffle(filename_list_bin)
random.shuffle(filename_list)


size = len(filename_list)
print(size)

# ghosts = np.loadtxt("/home/zerui603/KMTsimudata/bad_list.txt").astype(np.int)

# num_ghosts = len(ghosts)

def movefile(index):
	command = "mv"+" "+rootdir+filename_list[index]+" "+targetdir+str(int(index))+".npy"
	os.system(command)

def mvkillzero(index):
    
    try:
        data = list(np.load(rootdir+filename_list[index], allow_pickle=True))
        data_lc = list(np.array(data[1:],dtype=np.float64))
        labels = list(np.array(data[0],dtype=np.float64))

        lc_sig = np.array(data_lc[3])
        test_sig = 1/lc_sig

        data_array=data.copy()# np.array([labels,list(figdata)])
        data_array[0] = labels
        data_array = np.array(data_array)
        np.save(targetdir+str(406819+index)+".npy",data_array,allow_pickle=True)

        
        # print(datetime.datetime.now())
        del data_array
        return 0
            
    except:
        print(406819+index," Error")
        return 0
	

def examine(index):
    path_temp = "/scratch/zerui603/KMTsdfinal/val/"+str(index)+".npy"
    if os.path.exists(path_temp):
        pass
    else:
        print(str(index)," doesn't exist")

def judge_nan(index):
    path_temp = "/scratch/zerui603/KMTsdfinal/val/"+str(index)+".npy"
    data = np.load(path_temp,allow_pickle=True)
    data_lc = data[1:]
    num_nan = np.sum(np.isnan(data_lc[0]))
    if num_nan != 0:
        print("nan!",index)
    if len(data) != 5:
        print(index)

def mix(index):
    if index < bin_num:
        command = "cp"+" "+bindir+filename_list_bin[index]+" "+targetdir_mix+str(int(index))+".npy"
    else:
        command = "cp"+" "+sindir+filename_list_sin[index-bin_num]+" "+targetdir_mix+str(int(index))+".npy"
    os.system(command)
def mixval(index):
    if index < bin_num_val:
        command = "cp"+" "+bindir+filename_list_bin[int(-1*(index+1))]+" "+targetdir_mix_val+str(int(index))+".npy"
    else:
        command = "cp"+" "+sindir+filename_list_sin[int(-1*(index-bin_num_val+1))]+" "+targetdir_mix_val+str(int(index))+".npy"
    os.system(command)

examine_label_dir = ""
def examine_label(index):
    examine_label_dir

if __name__=="__main__":
    starttime = datetime.datetime.now()
    '''
    with mp.Pool(20) as p:
        p.map(mix, range(bin_num+sin_num))
    '''
    
    with mp.Pool(20) as p:
        p.map(mixval, range(bin_num_val+sin_num_val))
    endtime = datetime.datetime.now()
    
# 159000+146000

	
