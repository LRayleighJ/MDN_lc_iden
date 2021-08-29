import numpy as np
import scipy

def get_solution(x,y):
    sol=[]
    density = []
    for index in range(1,len(y)-2):
        dyleft = y[index+1]-y[index-1]
        dyright = y[index+2]-y[index]
        if dyright<=0 and dyleft>0:
            sol.append((x[index+1]+x[index])/2)
            density.append((y[index+1]+y[index])/2)
    return sol,density

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

def lc_hist(mag,time,sigma,num=200):

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

def gen_circle(x0,y0,radius,N=360):
    theta = np.linspace(0,2*np.pi,N)
    x = x0 + radius*np.cos(theta)
    y = y0 + radius*np.sin(theta)
    return x,y

def cal_alpha_0(ux,uy):
    cosvalue = ux/np.sqrt(ux**2+uy**2)
    sinvalue = uy/np.sqrt(uy**2+ux**2)
    alpha = np.arccos(cosvalue)
    
    if sinvalue < 0:
        alpha = 2*np.pi-alpha
    
    return alpha*180/np.pi

cal_alpha_np = np.frompyfunc(cal_alpha_0,2,1)

def cal_alpha(ux,uy):
    return cal_alpha_np(ux,uy).astype(np.float64)


def multi_gaussian_prob(pi,mu,sigma):
    def calc_multigaussian(y):
        prob = np.sum(pi/(np.sqrt(2*np.pi)*sigma)*np.exp(-1*(y-mu)**2/(2*sigma**2)))
        return prob
    return np.frompyfunc(calc_multigaussian,1,1)

def tran_lgq(lgq):
    return -1*lgq/4*5

def tran_lgs(lgs):
    return (lgs-np.log10(0.3))/(np.log10(3)-np.log10(0.3))*5

def tran_ui(ui):
    return (ui+1)/2*5

def detran_lgq(lgq_tran):
    return -1*lgq_tran*4/5

def detran_lgs(lgs_tran):
    return (lgs_tran/5)*(np.log10(3)-np.log10(0.3))+np.log10(0.3)

def detran_ui(ui_tran):
    return ui_tran/5*2-1

def tran_universal(data,index):
    if index == 0:
        return tran_lgq(data)
    if index == 1:
        return tran_lgs(data)
    if index == 2:
        return tran_ui(data)
    if index == 3:
        return tran_ui(data)
    else:
        return data

def magnitude_tran(magni,m_0=0):
    return m_0-2.5*np.log10(magni)