import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import random

name_group_list = ["00to10test","10to20test","20to30test","30to40test"]
size_test=500
name_group = ""
index_list = []
label_list = []
bspre_list = []
path_data = ""
path_store = "/scratch/zerui603/eyetest/"

def plotandmove(index):
    ## args: [u_0, rho, q, s, alpha, t_E, basis_m, t_0, chi^2, label]
    ## [args, times, dtimes, lc_noi, sigma, lc_nonoi, args_minimize, lc_fit_minimize, chi_array]
    global name_group
    global index_list
    global path_data
    posi = index_list[index]

    data_lc = np.load(path_data+str(index_list[index])+".npy",allow_pickle=True)
    args = data_lc[0]
    time = data_lc[1]
    lc_noi = data_lc[3]
    lc_err = data_lc[4]
    lc_nonoi = data_lc[5]
    lc_single = data_lc[7]
    chis_array = data_lc[8]

    mag_max_lim = np.mean(np.sort(lc_nonoi)[-25:])
    mag_min_lim = np.mean(np.sort(lc_nonoi)[:25])
    mag_max_lim += 0.1*(mag_max_lim-mag_min_lim)
    mag_min_lim -= 0.3*(mag_max_lim-mag_min_lim)

    plt.figure(figsize=(12,16))
    plt.subplot(211)
    plt.ylim(mag_min_lim,mag_max_lim)
    plt.scatter(time,lc_noi,s=4,alpha=0.5)
    plt.plot(time,lc_nonoi,ls="--",c="green")
    plt.plot(time,lc_single,ls="--",c="orange")
    plt.xlabel("t",fontsize=16)
    plt.ylabel("Mag",fontsize=16)
    plt.gca().invert_yaxis()
    plt.subplot(212)
    plt.scatter(time,np.array(chis_array)**2,s=4)
    plt.savefig(path_store+name_group[:-4]+"/fig/"+str(index)+".png")
    plt.close()

    os.system("cp "+path_data+"/"+str(index_list[index])+".npy "+path_store+name_group[:-4]+"/file/"+str(index)+".npy")

    print(name_group,index," has finished.")


if __name__=="__main__":
    for i in range(4):
        name_group = name_group_list[i]
        list_test = np.load("/home/zerui603/MDN_lc_iden/eyetest_index_"+name_group+".npy",allow_pickle=True)
        selectindex_list = random.sample(range(len(list_test[0])),500)
        print("SIZE: ",range(len(list_test[0])))
        selectindex_list = np.array(selectindex_list).astype(np.int)
        index_list = np.array(list_test[0])[selectindex_list].astype(np.int)
        label_list = np.array(list_test[1])[selectindex_list]
        bspre_list = np.array(list_test[2])[selectindex_list]
        path_data = "/scratch/zerui603/KMT_simu_lowratio/qseries/"+name_group+"/"
        with mp.Pool(20) as p:
            p.map(plotandmove,range(500))
        np.save("eyeafterselect_"+name_group+".npy",np.array([label_list,bspre_list]))
