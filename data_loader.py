# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:16:11 2019

@author: CNDLMembers
"""
import os
import shutil
import numpy as np
import pandas as pd

### loading raw AIF text files as a single numpy file: train, and test, respectively.
# put your AIF time-series data with 60 time-steps, obtained using NordicICE, and saved as text files in "raw_txt" directory.
    
raw_txt_path="./raw_txt"
num_patient = 386
test_size = 126

### AIF_DCE is saved as imgs_x1, x2 (measured twice), and AIF_DSC is saved as imgs_y1, y2 (measured twice)
# because the size of data is small (less than 10GB), usually it is more efficient to make the whole dataset as a single numpy file in this case.

file_list=os.listdir(raw_txt_path)

dce1_total=[]
dce2_total=[]
dsc1_total=[]
dsc2_total=[]

for file in sorted(map(int, file_list)): 
    print(file)
    dce1=np.loadtxt(raw_txt_path+"/"+str(file)+"/imgs_x1.txt")
    dce2=np.loadtxt(raw_txt_path+"/"+str(file)+"/imgs_x2.txt")
    dsc1=np.loadtxt(raw_txt_path+"/"+str(file)+"/imgs_y1.txt")
    dsc2=np.loadtxt(raw_txt_path+"/"+str(file)+"/imgs_y2.txt")
    
    dce1_total.append(dce1)
    dce2_total.append(dce2)
    dsc1_total.append(dsc1)
    dsc2_total.append(dsc2)

dce1_total=np.stack(dce1_total)
dce2_total=np.stack(dce2_total)
dsc1_total=np.stack(dsc1_total)
dsc2_total=np.stack(dsc2_total)

#%% test_set was randomly selected (n=126) from the index of total_set (n=386)

test_num=np.random.choice(range(0,num_patient), size=test_size)
np.savetxt("test_num.txt", test_num)
loaded_test_num=np.loadtxt("test_num.txt",dtype=int) 
test_num=loaded_test_num-1
train_num=np.array(list(set(range(0,num_patient)) - set(test_num))) 
train_num=np.sort(train_num)

dce1_train=dce1_total[train_num]
dce2_train=dce2_total[train_num]
dsc1_train=dsc1_total[train_num]
dsc2_train=dsc2_total[train_num]

dce1_test=dce1_total[test_num]
dce2_test=dce2_total[test_num]
dsc1_test=dsc1_total[test_num]
dsc2_test=dsc2_total[test_num]

x_train=np.concatenate((dce1_train, dce2_train), axis=0)
y_train=np.concatenate((dsc1_train, dsc2_train), axis=0)

x_test=np.concatenate((dce1_test, dce2_test), axis=0)
y_test=np.concatenate((dsc1_test, dsc2_test), axis=0)

# save as npy
np.save("x_train",x_train)
np.save("y_train",y_train)
np.save("x_test",x_test)
np.save("y_test",y_test)