import os
import multiprocessing as mp
import datetime
import numpy as np
import random

filename_list = os.listdir("/scratch/zerui603/KMT_simu_lowratio/single/")
random.shuffle(filename_list)
size = len(filename_list)
print(size)

# ghosts = np.loadtxt("/home/zerui603/KMTsimudata/bad_list.txt").astype(np.int)

# num_ghosts = len(ghosts)

def movefile(index):
	command = "mv /scratch/zerui603/KMT_simu_lowratio/single/"+filename_list[index]+" /scratch/zerui603/KMT_simu_lowratio/test/"+str(int(index+10000))+".npy"
	os.system(command)
	

def examine(index):
    path_temp = "/scratch/zerui603/KMTsdfinal/val/"+str(index)+".npy"
    if os.path.exists(path_temp):
        pass
    else:
        print(str(index)," doesn't exist")

root_killnan = "/scratch/zerui603/KMT_simu_lowratio/test/"
def judge_nan(index):
    path_temp = root_killnan+str(index)+".npy"
    data = np.load(path_temp,allow_pickle=True)
    data_lc = np.array(data[1:])
    #print([len(x) for x in data_lc])
    for i in range(5):
        num_nan = np.sum(np.isnan(data_lc[i]))
        if num_nan != 0:
            print("nan!",index,"posi: ",i)
            command = "cp "+root_killnan+str(index-1)+".npy "+root_killnan+str(index)+".npy"
            os.system(command)


if __name__=="__main__":
    starttime = datetime.datetime.now()
    with mp.Pool(20) as p:
        p.map(judge_nan, range(20000))
    endtime = datetime.datetime.now()
    
# 159000+146000

	
