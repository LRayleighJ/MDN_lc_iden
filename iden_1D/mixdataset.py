import numpy as np
import random
import os
import datetime
import multiprocessing as mp

name_group_list = ["00to05","05to10","10to15","15to20","20to25","25to30","30to35","35to40"]
name_group_test_list = ["00to05test","05to10test","10to15test","15to20test","20to25test","25to30test","30to35test","35to40test"]

rootdir = "/scratch/zerui603/KMT_simu_lowratio/qseries/"
targetdir = "/scratch/zerui603/KMT_simu_lowratio/qseries/mixval/"

size_eachorigin = 6250

number_binary = list(range(50000))
number_single = list(range(50000,100000))

clean_list = os.listdir(targetdir)

for filename in clean_list:
    os.system("rm -f "+targetdir+filename)

print("finish cleaning", datetime.datetime.now())

for i, name_group in enumerate(name_group_test_list):
    sample_binary_ori = random.sample(number_binary,size_eachorigin)
    sample_single_ori = random.sample(number_single,size_eachorigin)

    print(i,name_group,datetime.datetime.now())

    for j in range(size_eachorigin):
        command_binary = "cp "+rootdir+name_group+"/"+str(sample_binary_ori[j])+".npy"+" "+targetdir+str(i*size_eachorigin+j)+".npy"
        command_single = "cp "+rootdir+name_group+"/"+str(sample_single_ori[j])+".npy"+" "+targetdir+str(50000+i*size_eachorigin+j)+".npy"
        os.system(command_binary)
        os.system(command_single)

