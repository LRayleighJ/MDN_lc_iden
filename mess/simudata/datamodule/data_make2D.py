import numpy as np


def gaussian(x,sigma,mu):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))

def kill_escape(leftlim=0,rightlim=500-1):
    def kill_executer(x):
        if x < leftlim or x > rightlim:
            return -1
        else:
            return x
    return np.frompyfunc(kill_executer,1,1)

def kill_zero_0(x):
    if x<=0 :
        return 1
    else:
        return x

kill_zero_np = np.frompyfunc(kill_zero_0,1,1)

def inverse_zero_0(x):
    if x > 0:
        return 1/x
    else:
        return 0

inverse_zero_np = np.frompyfunc(inverse_zero_0,1,1)

def inverse_zero(x):
    return inverse_zero_np(x).astype(np.float64)

def kill_zero(x):
    return kill_zero_np(x).astype(np.float64)


def get_position(seq,leftlim,rightlim,dim = 500):
    cadence = (rightlim-leftlim)/dim
    return kill_escape(0,dim-1)((seq-leftlim)//cadence).astype(np.int64)

def get2Dmatrix(time,mag,sigma,mag_leftlim,mag_rightlim,dim = 500):
    sum_sigma_square_matrix = np.zeros((dim,dim))
    num_matrix = np.zeros((dim,dim))

    time_mean_cadence = (time[-1]-time[0])/(dim-1)

    time_position = get_position(time,time[0]-0.5*time_mean_cadence,time[-1]+0.5*time_mean_cadence,dim)
    mag_position = get_position(mag,mag_leftlim,mag_rightlim,dim)


    for index in range(len(time)):
        if time_position[index] >= 0 and mag_position[index] >= 0:
            sum_sigma_square_matrix[mag_position[index]][time_position[index]] += sigma[index]**2
            num_matrix[mag_position[index]][time_position[index]] += 1

    
    data_final = np.sqrt(sum_sigma_square_matrix/kill_zero(num_matrix))

    data_final = np.tanh(1/10*inverse_zero(data_final))

    return data_final

def get2Ddensity(time,mag,sigma,mag_leftlim,mag_rightlim,dim = 500):
    sum_sigma_square_matrix = np.zeros((dim,dim))
    num_matrix = np.zeros((dim,dim))

    time_mean_cadence = (time[-1]-time[0])/(dim-1)

    time_position = get_position(time,time[0]-0.5*time_mean_cadence,time[-1]+0.5*time_mean_cadence,dim)
    mag_position = get_position(mag,mag_leftlim,mag_rightlim,dim)
    mag_range = np.linspace(mag_leftlim,mag_rightlim,dim)


    for index in range(len(time)):
        if time_position[index] >= 0 :
            sum_sigma_square_matrix[time_position[index]] += gaussian(x=mag_range,sigma=sigma[index],mu=mag[index])

    data_final = sum_sigma_square_matrix.T

    return data_final