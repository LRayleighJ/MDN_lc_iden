import os
import multiprocessing as mp
import datetime
import numpy as np
import random

filename_list = os.listdir("/scratch/zerui603/KMTsimu2D_identify/training/")
random.shuffle(filename_list)
size = len(filename_list)
print(size)

# ghosts = np.loadtxt("/home/zerui603/KMTsimudata/bad_list.txt").astype(np.int)

# num_ghosts = len(ghosts)

def movefile(index):
	command = "mv /scratch/zerui603/KMTsimu2D_identify/training/"+filename_list[index]+" /scratch/zerui603/KMTsimu2D_identify/training_final/"+str(int(index)+460000)+".npy"
	os.system(command)
	

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

if __name__=="__main__":
    starttime = datetime.datetime.now()
    with mp.Pool() as p:
        p.map(movefile, range(400000))
    endtime = datetime.datetime.now()
    
# 159000+146000

	
