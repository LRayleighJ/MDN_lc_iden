import numpy as np

def gridcount2D(x_data,y_data,X,Y):
    if x_data.shape != y_data.shape:
        raise RuntimeError("The shapes of x_data and y_data don't match")
    data = np.ones(len(x_data))
    '''
    delta_x = (X[-1]-X[0])/(len(X)-1)
    delta_y = (Y[-1]-Y[0])/(len(Y)-1)
    x_bin = np.linspace(X[0]+0.5*delta_x,X[-1]-0.5*delta_x,len(X)-1)
    y_bin = np.linspace(Y[0]+0.5*delta_y,Y[-1]-0.5*delta_y,len(Y)-1)
    '''
    x_bin = []
    y_bin = []
    for i in range(len(X)-1):
        x_bin.append((X[i+1]+X[i])/2)
    for j in range(len(Y)-1):
        y_bin.append((Y[j+1]+Y[j])/2)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    count_matrix = np.ones((len(x_bin),len(y_bin),))
    for i in range(len(x_bin)):
        for j in range(len(y_bin)):
            data_bin = data[(x_data>X[i])&(x_data<=X[i+1])&(y_data>Y[j])&(y_data<=Y[j+1])]
            count_matrix[i][j] = len(data_bin)  

    return x_bin,y_bin,count_matrix

def getborder(y_mat): # y_mat:(x,y)
    border_array = []

    rate_list = np.array([0.25,0.5,0.75])

    x_len = y_mat.shape[0]
    y_len = y_mat.shape[1]

    for i in range(x_len):
        border_x = list(-1*np.ones(rate_list.shape))
        y_x = y_mat[i]
        for index_rate in range(len(rate_list)):
            rate = rate_list[index_rate]
            for j in range(y_len-1):
                if (y_x[j]<rate)&(y_x[j+1]>=rate):
                    border_x[index_rate] = j
                    break
        border_array.append(border_x)
    return border_array


