# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:53:04 2019

@author: CNDLMembers
"""
import matplotlib.pyplot as plt
from scipy import signal
import os
import numpy as np
import shutil
from keras.models import *
from keras.utils import *
from keras.layers import *
from utils import *

#%% load two separate test sets
x_test=np.load("Zxx_test.npy") 
y_test=np.load("Zyy_test_poly.npy") # (None,8,10,2)

x1_test=x_test[:len(x_test)//2] # AIF_DCE1
x2_test=x_test[len(x_test)//2:] # AIF_DCE2

y1_test=y_test[:len(y_test)//2] # AIF_DSC1
y2_test=y_test[len(y_test)//2:] # AIF_DSC2

#%% plot fake_y over imgs_x, y

opt_generator=load_model('generator_poly.h5') 
test_num_file="test_num.txt"

imgs_x1_plot_test, imgs_y1_plot_test, fake_y1_plot_test=test_plot_generator(x1_test, y1_test, opt_generator, test_num_file)
imgs_x2_plot_test, imgs_y2_plot_test, fake_y2_plot_test=test_plot_generator(x2_test, y2_test, opt_generator, test_num_file)

#%% resample back to original for generated DSC: (0,240,60) -> (0,90,60)

fake_y1_test_resampled_back=resample_y_back(fake_y1_plot_test, choose=1) #(None,60) but 0-240 -> (None,60) for 0-90
fake_y2_test_resampled_back=resample_y_back(fake_y2_plot_test, choose=1) #(None,60) but 0-240 -> (None,60) for 0-90

# save as npy file
np.save("fake_y1_test_resampled_back.npy", fake_y1_test_resampled_back)
np.save("fake_y2_test_resampled_back.npy", fake_y2_test_resampled_back)

#%% copy fake_y to corresponding directories with specified suffix of filename 

suffix="poly"
path_plot1="fake_y1_plot_test_"+suffix
path_plot2="fake_y2_plot_test_"+suffix

copy_resampled_y_test_only(path_plot1, fake_y1_plot_test, test_num_file, suffix)
copy_resampled_y_test_only(path_plot2, fake_y2_plot_test, test_num_file, suffix)

