import numpy as np
import MulensModel as mm
import os
import matplotlib.pyplot as plt
import datetime
import random
import multiprocessing as mp
import gc
import sys
import traceback
import scipy.optimize as op
from scipy.optimize import curve_fit

# get arguments
## bins of mag: 8 ; bins of lgdchi: 8
magnitude_bins = np.linspace(18,22,9)
lgdchis_bins = np.linspace(1,5,9)

# get location of bins
posi_mag = np.int(sys.argv[1])
posi_lgdchis = np.int(sys.argv[2])

basis_m_min = magnitude_bins[posi_mag]
basis_m_max = magnitude_bins[posi_mag+1]
lgdchis_min = lgdchis_bins[posi_lgdchis]
lgdchis_max = lgdchis_bins[posi_lgdchis+1]

datadir_time = "/scratch/zerui603/noisedata/timeseq/"
datadir_noise = "/scratch/zerui603/noisedata/noisedata_hist/"

storedir = "/scratch/zerui603/KMT_simu_lowratio/qseries/10to15testbins/"+str(posi_mag)+"_"+str(posi_lgdchis)+"/"

os.system("mkdir "+storedir)

# number range: range(num_echo*num_batch*num_bthlc, (num_echo+1)*num_bthlc)

singleorbinary = 1 # 0:single 1:binary, else: error

num_echo = 0
num_batch = 40
num_bthlc = 25
num_process = 16

def generate_random_parameter_set(u0_max=1, max_iter=100):
    ''' generate a random set of parameters. '''
    rho = 10.**random.uniform(-4, -2) # log-flat between 1e-4 and 1e-2
    q = 10.**random.uniform(-1.5, -1.) # including both planetary & binary events
    s = 10.**random.uniform(np.log10(0.3), np.log10(3))
    alpha = random.uniform(0, 360) # 0-360 degrees
    ## use Penny (2014) parameterization for small-q binaries ##
    if q < 1e-3:
        if q/(1+q)**2 < (1-s**4)**3/27/s**8: # close topology #
            if s < 0.1:
                uc_max = 0
            else:
                uc_max = (4+90*s**2)*np.sqrt(q/(1+s**2))/s
            xc = (s-(1-q)/s)/(1+q)
            yc = 0.
        elif s**2 > (1+q**(1/3.))**3/(1+q): # wide topology #
            uc_max = (4+min(90*s**2, 160/s**2))*np.sqrt(q)
            xc = s - 1./(1+q)/s
            yc = 0.
        else: # resonant topology
            xc, yc = 0., 0.
            uc_max = 4.5*q**0.25
        alpha_rad = alpha/180.*np.pi
        n_iter = 0
        for i_3 in range(50):
            uc = random.uniform(0, uc_max)
            u0 = uc - xc*np.sin(alpha_rad) + yc*np.cos(alpha_rad)
            n_iter += 1
            if u0 < u0_max:
                break
            if n_iter > max_iter:
                break
    else: # for large-q binaries, use the traditional parameterization
        u0 = random.uniform(0, u0_max)
    return (u0, rho, q, s, alpha)

class TimeData(object):
    def __init__(self,datadir,num_point,bad_seq_threshold=35):
        self.time_datadir = datadir
        time_file_list = os.listdir(datadir)
        time_num_file = len(time_file_list)
        time_index_file = random.randint(0,time_num_file-1)
        self.filename = time_file_list[time_index_file]
        self.time_data = list(np.load(datadir+time_file_list[time_index_file],allow_pickle=True))
        self.num_timeseq = len(self.time_data) 
        self.num_point = num_point

        self.time_data_available = [x for x in range(len(self.time_data)) if len(self.time_data[x]) > num_point]
        self.num_available = len(self.time_data_available)
        self.bad_sequence_threshold = bad_seq_threshold
    
    def delta_t(self,T):
        T_forward = np.append(T,0)
        T_backward = np.append(2*T[0]-T[1],T)
        T_ori = T_forward-T_backward
        return T_ori[:-1]

    def badseq(self,seq,limit):
        jud_seq = seq-limit
        jud_seq = jud_seq + np.abs(jud_seq)
        if np.sum(jud_seq)>0:
            return 1
        else:
            return 0

    def get_t_seq(self,max_count=20):
        count = 0
        
        for i_1 in range(50):
            index_timeseq = 0
            count_gettimeseq = 0
            for i_2 in range(50):
                index_timeseq = random.randint(0,self.num_available-1)
                realtimeseq = self.time_data[self.time_data_available[index_timeseq]]
                if len(realtimeseq)>self.num_point:
                    realtimeseq = np.sort(realtimeseq)
                    break
                else:
                    print("not enough length.",self.filename,len(realtimeseq))
                    count_gettimeseq += 1
                    if count_gettimeseq > max_count:
                        raise RuntimeError("Very bad data, code destoryed")
                    continue
            index_cut = random.randint(self.num_point,len(realtimeseq)-1)
            time_cut = realtimeseq[index_cut-self.num_point:index_cut]-np.mean(realtimeseq[index_cut-self.num_point:index_cut])

            d_times = self.delta_t(time_cut)
            count += 1
            if count >= max_count:
                raise RuntimeError("Max step num has reached.")
                
            
            if self.badseq(d_times,self.bad_sequence_threshold*np.mean(d_times)):
                # print("Time cadence is not well.",count)
                continue
            else:
                # print("successfully get timedata")
                return time_cut,d_times
    def __del__(self):
        del self.time_datadir
        del self.filename
        del self.time_data
        del self.num_timeseq
        del self.num_point

        del self.time_data_available
        del self.num_available
        gc.collect()        
                
            
class NoiseData(object):
    def __init__(self,datadir):
        self.datadir = datadir
        file_list = os.listdir(datadir)
        self.num_file = len(file_list)
        index_file = random.randint(0,self.num_file-1)
        noise_data = list(np.load(datadir+file_list[index_file],allow_pickle=True))
        self.mag_range = noise_data[0]
        self.state = noise_data[1]
        self.err_state = noise_data[2]

    
    def gen_index(self,mag,cad=0.1):     
        return ((np.array(mag)-self.mag_range[0])//cad).astype(np.int)

    def noisemodel(self,mag):
        posi = self.gen_index(mag).astype(np.int)
        if np.min(posi)<0 or np.max(posi)>=len(self.state):
            print("mag range error: ",np.min(posi),np.max(posi),len(self.state))
            
            raise RuntimeError("The range of magnitude is out of range")
        errorbar = []
        for i_posi in range(len(posi)):
            histbin_index = np.int(np.random.rand()*self.state[posi[i_posi]])
            errorbar.append(self.err_state[posi[i_posi]][histbin_index])

        errorbar = np.array(errorbar)
        gauss_list = np.random.randn(len(mag))

        return gauss_list*errorbar,errorbar

    
    def __del__(self):
        del self.datadir 
        del self.mag_range
        del self.state
        del self.err_state
        gc.collect()


def magnitude_tran(magni,m_0=20):
    return m_0 - 2.5*np.log10(magni)

def mag_cal(t,tE,t0,u0,m0):
    u = np.sqrt(((t-t0)/tE)**2+u0**2)
    A = (u**2+2)/(u*np.sqrt(u**2+4))
    return m0-2.5*np.log10(A)

def chi_square(single,binary,sigma):
    return np.sum(np.power((single-binary)/sigma,2))

def chi2_for_minimize(args,time,mag,sigma):
    return chi_square(mag_cal(time,*args),mag,sigma)


def gen_simu_data(index_batch):
    print("Batch",index_batch,"has started")
    c_time = TimeData(datadir=datadir_time,num_point=2000)
    noi_model = NoiseData(datadir=datadir_noise)
    counter_total = 0
    for index_slc in range(num_bthlc):
        counter_total = 0
        
        for i_5 in range(50):
            try:
                times,d_times = c_time.get_t_seq()
                t_E = (times[-1]-times[0])/8
                if t_E <= 0.:
                    raise RuntimeError("tE<=0")
                break
            except:
                del c_time
                gc.collect()
                c_time = TimeData(datadir=datadir_time,num_point=2000)
                print("time failure once")
                continue

        if i_5 == 49:
            print(str(index_batch*num_bthlc+index_slc), "time-getting failure")
        t_E = (times[-1]-times[0])/8

        # for test 

        times_fortest = times[500:500+1000]

        t_0 = 0
        count_gen_args = 0
        count_args_bearing = 0
        while True:
            try:
                basis_m = random.uniform(basis_m_min,basis_m_max)
                u_0, rho, q, s, alpha = generate_random_parameter_set()
                if u_0 == 0:
                    while True:
                        u_0, rho, q, s, alpha = generate_random_parameter_set()
                        if u_0 != 0:
                            break
                        else:
                            continue
                args_data_test = [u_0, rho, q, s, alpha, t_E, basis_m, t_0]
                # single = mm.Model({'t_0': t_0, 'u_0': u_0, 't_E': t_E})
                if singleorbinary == 1:
                    planet = mm.Model({'t_0': t_0, 'u_0': u_0, 't_E': t_E, 's': s, 'q': q, 'alpha': alpha,'rho': rho})
                elif singleorbinary == 0:
                    planet = mm.Model({'t_0': t_0, 'u_0': u_0, 't_E': t_E})
                else:
                    raise RuntimeError("You should give singleorbinary the value 0 or 1.")
                # planet_degeneracy = mm.Model({'t_0': t_0, 'u_0': u_0, 't_E': t_E, 's': 1/s, 'q': q, 'alpha': alpha,'rho': rho})
                
                planet.set_default_magnification_method('VBBL')
                
                lc = planet.magnification(times)

                if np.min(magnitude_tran(lc,m_0=basis_m)) < 9.3:
                    continue

                noi, sig = noi_model.noisemodel(magnitude_tran(lc,m_0=basis_m))
                lc_noi = magnitude_tran(lc,basis_m) + noi

                lc_noi_fortest = lc_noi[500:500+1000]
                sig_fortest = sig[500:500+1000]

                # Minimize using extend fitting
                ## [u_0, rho, q, s, alpha, t_E, basis_m, t_0]
                initial_guess = [t_E,t_0,u_0,basis_m]

                try:
                    result = op.minimize(chi2_for_minimize, x0=initial_guess,args=(times_fortest,lc_noi_fortest,sig_fortest), method='Nelder-Mead')
                except:
                    print("Minimize error")
                    continue

                # print("Fitting was successful? {:}".format(result.success))
                if not result.success:
                    print(result.message)
                # print("Function evaluations: {:}".format(result.nfev))
                if isinstance(result.fun, np.ndarray):
                    if result.fun.ndim == 0:
                        result_fun = float(result.fun)
                    else:
                        result_fun = result.fun[0]
                else:
                    result_fun = result.fun

                args_minimize_fortest = result.x.tolist()

                lc_fit_minimize_fortest = mag_cal(times_fortest,*args_minimize_fortest)
                chi_s_minimize_fortest = chi_square(lc_fit_minimize_fortest,lc_noi_fortest,sig_fortest)
                
                chi_s_model = chi_square(magnitude_tran(lc[500:500+1000],basis_m),lc_noi_fortest,sig_fortest)

                delta_chi_s_fortest = chi_s_minimize_fortest-chi_s_model  

                if (delta_chi_s_fortest > 10**lgdchis_max)|(delta_chi_s_fortest < 10**lgdchis_min):
                    #print("delta chi square fails once")
                    continue
                
                break
            
            except:
                print(traceback.print_exc())
                print("fail generation of args")
                count_gen_args += 1
                if count_args_bearing > 5:
                    raise RuntimeError("Noise model crash")
                    break
                if count_gen_args > 20:
                    print("Reload noise model: ", count_args_bearing)
                    noi_model = NoiseData(datadir=datadir_noise)
                    count_gen_args = 0
                    count_args_bearing += 1
                continue
        # original
        ## [u_0, rho, q, s, alpha, t_E, basis_m, t_0, chi^2,chi^2_test, label]
        ## [times, dtimes, lc_noi, sigma, lc_nonoi, args_minimize, lc_fit_minimize,args_minimize_test, lc_fit_minimize_test, chi_array]
        # the data now we need
        ## [u_0, rho, q, s, alpha, t_E, basis_m, t_0, chi^2, label]
        ## [times, dtimes, lc_noi, sigma, lc_nonoi, args_minimize, lc_fit_minimize, chi_array]
        args_data_test.append(delta_chi_s_fortest)
        args_data_test.append(singleorbinary)
        
        data_array=np.array([args_data_test,list(times_fortest),list(d_times[500:500+1000]),list(lc_noi_fortest),list(sig_fortest),list(magnitude_tran(lc[500:500+1000],basis_m)),list(args_minimize_fortest),list(lc_fit_minimize_fortest),list((lc_noi_fortest-lc_fit_minimize_fortest)/sig_fortest)],dtype=object)
        np.save(storedir+str(index_batch*num_bthlc+index_slc)+".npy",data_array,allow_pickle=True)
        # print("lc "+str(index_batch*num_bthlc+index_slc),datetime.datetime.now())
    
    del c_time
    del noi_model
    gc.collect()
        
if __name__=="__main__":
    
    starttime = datetime.datetime.now()
    print("starttime:",starttime)
    print("cpu_count:",os.cpu_count())

    with mp.Pool(num_process) as p:
        p.map(gen_simu_data, range(0,num_batch))
        
    endtime = datetime.datetime.now()
    print("end time:",endtime)
    print("total:",endtime - starttime)