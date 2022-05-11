import numpy as np
import scipy

def kill_zero(x):
    if x == 0:
        return 1
    else:
        return x

kill_zero_np = np.frompyfunc(kill_zero,1,1)

def k_zero(x):
    return kill_zero_np(x).astype(np.float64)

def judge_value_0(x,limit=1):
    if x > limit:
        return 1
    else:
        return 0

jv_np = np.frompyfunc(judge_value_0,1,1)

def judge_value(x):
    return jv_np(x).astype(np.float64)

def lc_hist(mag,time,sigma,num=500):

    time_range = (time[-1]-time[0])*(num+1)/num
    cadence = time_range/num
    left_limit = time[0]-0.5*cadence

    posi_list = (np.array(time)-left_limit)//cadence
    posi_list = posi_list.astype(np.int)

    value_list = np.zeros(num)
    weight_list = np.zeros(num)
    num_list = np.zeros(num)

    len_ori = len(time)


    for i in range(len_ori):
        index_posi = posi_list[i]
        if sigma[i] == 0:
            print("0 in sigma")
            continue
        value_list[index_posi] += mag[i]/(1000*sigma[i])**2
        weight_list[index_posi] += 1/(1000*sigma[i])**2
        num_list[index_posi] += 1

    data_histed = value_list/k_zero(weight_list)
    time_ref = np.linspace(time[0],time[-1],len(data_histed))

    time_killsingle = []
    data_killsingle = []

    for i in range(len(data_histed)):
        if num_list[i] - 1 <=0:
            continue
        else:
            time_killsingle.append(time_ref[i])
            data_killsingle.append(data_histed[i])

    data_final = np.interp(time_ref,time_killsingle,data_killsingle)
    return np.array(data_final)