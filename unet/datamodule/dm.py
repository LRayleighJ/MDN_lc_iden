import numpy as np

def get_rateborder(rate,rateup,datax,datay,xbins):
    datax = np.array(datax)
    datay = np.array(datay)
    upbound_list = []
    downbound_list = []
    xpoint_list = []
    for i in range(len(xbins)-1):
        data_bins = datay[(datax<xbins[i+1])&(datax>xbins[i])]
        if len(data_bins) < 10:
            continue
        xpoint = (xbins[i+1]+xbins[i])/2
        upbound = np.sort(data_bins)[np.min([np.int((1-rateup)*len(data_bins)),len(data_bins)-1])]
        downbound = np.sort(data_bins)[np.int((1-rate)*len(data_bins))]
        upbound_list.append(upbound)
        downbound_list.append(downbound)
        xpoint_list.append(xpoint)
    
    return np.array(upbound_list), np.array(downbound_list), np.array(xpoint_list)

def get_up_down_border(data,rate,central):
    value_mean = central

    data_halfup = np.sort(data[data>value_mean])
    data_halfdown = np.sort(data[data<value_mean])

    if len(data_halfup) < 1:
        border_up = value_mean
    else:
        border_up = data_halfup[np.min([len(data_halfup)-1,np.int(rate*len(data_halfup))])]

    if len(data_halfdown) < 1:
        border_down = value_mean
    else:
        border_down = data_halfdown[np.min([len(data_halfdown)-1,np.int((1-rate)*len(data_halfdown))])]
    return border_down,border_up

def get_rate_updown_line(rate,datax,datay,xbins):
    datax = np.array(datax)
    datay = np.array(datay)
    upbound_list = []
    downbound_list = []
    xpoint_list = []
    for i in range(len(xbins)-1):
        data_bins = datay[(datax<xbins[i+1])&(datax>xbins[i])]
        if len(data_bins) < 10:
            continue
        xpoint = (xbins[i+1]+xbins[i])/2
        downbound, upbound = get_up_down_border(data_bins, rate, np.mean(data_bins))
        upbound_list.append(upbound)
        downbound_list.append(downbound)
        xpoint_list.append(xpoint)
    # print(upbound_list.shape, downbound_list.shape, xpoint_list.shape)
    return np.array(upbound_list), np.array(downbound_list), np.array(xpoint_list)


