import numpy as np

def gridcount2D(x_data,y_data,X,Y):
    print(x_data.shape)
    if x_data.shape != y_data.shape:
        raise RuntimeError("The shapes of x_data and y_data don't match")
    data = np.ones(len(x_data))
    delta_x = (X[-1]-X[0])/(len(X)-1)
    delta_y = (Y[-1]-Y[0])/(len(Y)-1)
    x_bin = np.linspace(X[0]+0.5*delta_x,X[-1]-0.5*delta_x,len(X)-1)
    y_bin = np.linspace(Y[0]+0.5*delta_y,Y[-1]-0.5*delta_y,len(Y)-1)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    count_matrix = np.ones((len(x_bin),len(y_bin),))
    for i in range(len(x_bin)):
        for j in range(len(y_bin)):
            data_bin = data[(x_data>X[i])&(x_data<X[i+1])&(y_data>Y[j])&(y_data<Y[j+1])]
            count_matrix[i][j] = len(data_bin)  

    return x_bin,y_bin,count_matrix

