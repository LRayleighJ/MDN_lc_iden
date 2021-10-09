import urllib.request
import re
import os
import numpy as np
import pandas as pd


def grab(url):
    # 打开传入的网址
    resp = urllib.request.urlopen(url)
    # 读取网页源码内容
    data = resp.read().decode('utf-8')
    labels = ["t_0","[t_E]*[t_eff]*",r"u_0"]
    values = []
    for label in labels:
        pattern = r"<td width=130>"+label+"</td>.*</tr>"
        te_str = re.findall(pattern, data)[0]
        match_te = re.findall(r"<td>[\.\d]*</td>",te_str)
        match_te_f = re.split(r"[<>]",match_te[0])[2]
        values.append(float(match_te_f))
    return values


targetdir = "/scratch/zerui603/KMTData/"

years = [2016,2017,2018,2019,2020]
num_events = [2588, 2817, 2781, 3303, 894]

frame_label = ["t_0","t_E","u_0"]

frame_array = []
frame_index = []

for kk in range(1,5):
    for i in range(1,num_events[kk]+1):
        try:
            file_number = "%04d" % i
            folder_name = str(years[kk])+ '_' + file_number
            event_str = "KMT-%04d-BLG-%04d"%(years[kk],i)
            print(event_str)
            url_event = 'http://kmtnet.kasi.re.kr/~ulens/event/'+str(years[kk])+'/view.php?event='+event_str
            frame_array.append(grab(url_event))
            frame_index.append(folder_name)
            print(folder_name," has completed")
        except:
            print(years[kk],"%04d" % i,"DAMN ERROR")
            continue

frame_array = np.array(frame_array)
df_array = pd.DataFrame(frame_array,columns=frame_label,index=frame_index)

recarray = df_array.to_records()
np.save("KMT_args", recarray)



