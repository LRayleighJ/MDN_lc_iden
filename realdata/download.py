import os
import numpy as np
import multiprocessing as mp

targetdir = "/scratch/zerui603/KMTrealdata/"

years = ['2019']
num_events = [3303]

num_total = np.sum(num_events)

def get_years(index):
    judge = index
    for i in range(len(num_events)):
        if judge - num_events[i] < 0:
            break
        judge -= num_events[i]
    return i,judge

def download(index):
    kk,i = get_years(index)
    i_blyat = i+1
    file_number = "%04d" % i_blyat
    folder_name = years[kk]+ '_' + file_number
    file_name = 'http://kmtnet.kasi.re.kr/~ulens/event/'+years[kk]+'/data/KB'+years[kk][2:]+str(file_number)+'/pysis/pysis.tar.gz'
    os.system("mkdir " + targetdir + folder_name)
    os.system("wget -O "+ targetdir + "temp_"+str(kk)+"_"+str(i)+".tar.gz " +file_name)
    os.system("tar -xvzf "+targetdir+"temp"+str(kk)+"_"+str(i)+".tar.gz -C "+ targetdir + folder_name)
    os.system("rm -f "+targetdir+"temp"+str(kk)+"_"+str(i)+".tar.gz")
    print(folder_name," has completed")
    return 0


if __name__=="__main__":
    print("cpu_count:",os.cpu_count())
    with mp.Pool(6) as p:
        p.map(download, range(num_total))
    


