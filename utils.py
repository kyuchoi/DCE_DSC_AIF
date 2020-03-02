# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:08:06 2019

@author: CNDLMembers
"""

import matplotlib.pyplot as plt
from scipy import signal, interpolate
from scipy.interpolate import spline
from keras.models import *
from sklearn.utils import shuffle
import os
import shutil
import glob
import numpy as np
import pandas as pd
from random import gauss
import openpyxl

#%% preprocessing 
    
# add_steps 3 (60 -> 63)
def x_add_steps(x_total, add_steps): 
    x_total_list=[]
    for i in range(len(x_total)):
        x_total_i=np.pad(np.squeeze(x_total[i]), (0,add_steps), 'edge')
        x_total_list.append(x_total_i)
    x_total=np.stack(x_total_list)
    return x_total

# stft
def x_stft(x_total, y_total, nperseg=14, noverlap=7):
    Zxx_total=[]
    Zyy_total=[]
    for idx in range(x_total.shape[0]):
        x=x_total[idx]
        y=y_total[idx]
        f_x,t_x,Zxx=signal.stft(x,nperseg=nperseg, noverlap=noverlap)
        f_y,t_y,Zyy=signal.stft(y,nperseg=nperseg, noverlap=noverlap)
        Zxx_total.append(Zxx)
        Zyy_total.append(Zyy)
    Zxx_total=np.stack(Zxx_total)
    Zyy_total=np.stack(Zyy_total)
    return Zxx_total, Zyy_total

# add noise to augment
def x_add_noise(x_total, y_total, avg=0.0, std=0.2):
    x_n_total=[]
    y_n_total=[]
    for idx in range(x_total.shape[0]):
        x=x_total[idx]
        y=y_total[idx]
        
        noise=[gauss(avg, std) for i in range(x_total.shape[1])] #(60,) dim noise generation
        x_n=x+noise
        y_n=y+noise
        x_n_total.append(x_n)
        y_n_total.append(y_n)
    x_n_total=np.stack(x_n_total)
    y_n_total=np.stack(y_n_total)
    return x_n_total, y_n_total

# augment using flipping (only along horizontal)
def x_flip_lr(x, y):
    x_lr=np.flip(x, axis=1)
    y_lr=np.flip(y, axis=1)
    return x_lr, y_lr

# make complex number into 2 channels
def complex2chan(Zxx_total_aug, Zyy_total_aug):
    # for x_total
    x_total_fin_tot=[]
    for idx in range(Zxx_total_aug.shape[0]):
        x_total_fin=np.concatenate((np.expand_dims(Zxx_total_aug[idx].real, axis=-1), np.expand_dims(Zxx_total_aug[idx].imag, axis=-1)), axis=-1)
        x_total_fin_tot.append(x_total_fin)    
    Zxx_total_aug_chan=np.stack(x_total_fin_tot)    
    
    # for y_total
    y_total_fin_tot=[]
    for idx in range(Zyy_total_aug.shape[0]):
        y_total_fin=np.concatenate((np.expand_dims(Zyy_total_aug[idx].real, axis=-1), np.expand_dims(Zyy_total_aug[idx].imag, axis=-1)), axis=-1)
        y_total_fin_tot.append(y_total_fin)    
    Zyy_total_aug_chan=np.stack(y_total_fin_tot)    

    return Zxx_total_aug_chan, Zyy_total_aug_chan

#%% utils for fake_y_generator

# resample AIF_DSC to match AIF_DCE for timescale

def resample_y(y_total, choose, plot): 
    y_total_resampled=[]
    time_dsc = np.linspace(0, 90, 60, endpoint=False) 
    time_dsc_pad = np.linspace(0, 240, 160, endpoint=False)
    time_dce = np.linspace(0, 240, 60, endpoint=False)
    
    for i in range(y_total.shape[0]):
        y=y_total[i] 
        add_steps=100 
        y_add=np.tile(y[-1],add_steps) 
        y_pad=np.concatenate((y,y_add),axis=0) # timestep=160
    
        if choose:
            tck = interpolate.splrep(time_dsc_pad, y_pad, s=0)
            y_spl = interpolate.splev(time_dce, tck, der=0) # timestep=60, but 0-240
            y_total_resampled.append(y_spl)
        else:
            y_resampled_poly = signal.resample_poly(y_pad,60,160)
            y_total_resampled.append(y_resampled_poly)
        # plot original vs resampled
        if plot:
            fig,ax=plt.subplots(2,figsize=(5,5))
            ax[0].plot(time_dsc, y, label="dsc", color='blue')
            ax[1].plot(time_dce, y_spl, label="dsc_pad_rec", color='orange')
            fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()
    y_total_resampled=np.stack(y_total_resampled)
    return y_total_resampled 

# resample back to 60 from 240

def resample_y_back(y_resampled_total, choose): 
    y_total_original=[]
    original_length=60
    time_dsc = np.linspace(0, 90, 60, endpoint=False) 
    time_dsc_pad = np.linspace(0, 240, 160, endpoint=False)
    time_dce = np.linspace(0, 240, 60, endpoint=False)
    
    y_resampled_total=y_resampled_total[:,:original_length]
    for i in range(y_resampled_total.shape[0]):
        y=y_resampled_total[i] 
        
        if choose:
            tck_back = interpolate.splrep(time_dce, y, s=0)
            y_spl_back = interpolate.splev(time_dsc_pad, tck_back, der=0)
            y_back_spl=y_spl_back[:original_length]
            y_total_original.append(y_back_spl)
        else:
            y_back_poly = signal.resample_poly(y,160,60)[:original_length]
            y_total_original.append(y_back_poly)
    y_total_original=np.stack(y_total_original)
    return y_total_original 
    
# copy resampled y into directories of the test set
    
def copy_resampled_y_test_only(path_plot, y_total_resampled, test_num_file, suffix): # "suffix" defines suffix of the filename
    path="raw_data" # raw_data contains the DCE-MRI dicom files, and tumor ROI nifti file (.nii) in the directory for each patient 
    test_num=np.loadtxt(test_num_file)
    if not os.path.isdir(path_plot):
        os.mkdir(path_plot)
        
    for i in range(y_total_resampled.shape[0]):
        print(path_plot+'/'+path_plot[:7]+'_plot_total_%d.txt' % (i+1), y_total_resampled[i].shape)    
        np.savetxt(path_plot+'/'+path_plot[:7]+'_plot_total_%d.txt' % (i+1), y_total_resampled[i])
    plot_list=os.listdir(path_plot)
    
    for plot in sorted(plot_list):
        idx_plot=plot.split('_')[4].split('.')[0]
        idx_plot=int(idx_plot)
        idx=int(test_num[idx_plot-1])
        print(path_plot+"/"+plot, path+"/"+str(idx)+"/"+plot[:7]+"_resampled_"+str(suffix)+".txt")
        shutil.copy(path_plot+"/"+plot, path+"/"+str(idx)+"/"+plot[:7]+"_resampled_"+str(suffix)+".txt")

# plot function for test set
        
def test_plot_generator(x_test, y_test, opt_generator, test_num_file): 
    imgs_x_plot_total=[]
    imgs_y_plot_total=[]
    fake_y_plot_total=[]
    test_num_plot=np.loadtxt(test_num_file)
    
    for id, (imgs_x, imgs_y) in enumerate(zip(x_test, y_test)):
        imgs_x=imgs_x[np.newaxis,:,:,:] # (8,10,2) -> (1,8,10,2)
        imgs_y=imgs_y[np.newaxis,:,:,:]
        fake_y = opt_generator.predict(imgs_x) 
        
        t_rec, fake_y_plot=signal.istft(fake_y[0,:,:,0]+1j*fake_y[0,:,:,1], nperseg=14)
        t_rec, imgs_y_plot=signal.istft(imgs_y[0,:,:,0]+1j*imgs_y[0,:,:,1], nperseg=14)            
        t_rec, imgs_x_plot=signal.istft(imgs_x[0,:,:,0]+1j*imgs_x[0,:,:,1], nperseg=14)
        
        imgs_x_plot_total.append(imgs_x_plot)
        imgs_y_plot_total.append(imgs_y_plot)
        fake_y_plot_total.append(fake_y_plot)
        
        # plot time-signal curve
        print("patient ID:",int(test_num_plot[id]),"id:",id)
        fig, ax=plt.subplots(1,3, figsize=(15,2))
        
        ax[0].plot(t_rec, fake_y_plot, label='generated DSC')
        ax[0].plot(t_rec, imgs_y_plot, label='DSC')
        ax[0].plot(t_rec, imgs_x_plot, label='DCE')
        ax[0].set_xlabel("Time-steps(TR)")
        ax[0].set_ylabel("Signal intensity")
        
        # plot spectrogram
        ax[1].imshow(np.sqrt(fake_y[0,:,:,0]**2+fake_y[0,:,:,1]**2))
        ax[2].imshow(np.sqrt(imgs_y[0,:,:,0]**2+imgs_y[0,:,:,1]**2))
        plt.show()
    
    imgs_x_plot_test=np.stack(imgs_x_plot_total)
    imgs_y_plot_test=np.stack(imgs_y_plot_total)
    fake_y_plot_test=np.stack(fake_y_plot_total)
    
    return imgs_x_plot_test, imgs_y_plot_test, fake_y_plot_test

#%% etc utils

# compute PSNR for dataset

import math

def compute_single_PSNR(y_pred, y_true):
    y_pred = y_pred.astype(np.float64)
    y_true = y_true.astype(np.float64)
    y_true_max = np.max(y_true)
    mse = np.mean((y_pred - y_true) ** 2)
    if mse == 0:
        return "Same Image"
    return 20 * math.log10(y_true_max) - 10 * math.log10(1. / mse) 

def compute_total_PSNR(X_total, Y_total, batch_size):    
    total_psnr = []
    for batch_i, (image_batch_x, image_batch_y) in load_batch(X_total, Y_total, batch_size):
        generated_batch_y = generator.predict(image_batch_x)
        batch_psnr = []
        for image_y, generated_y in zip(image_batch_y, generated_batch_y):
            single_psnr = compute_single_PSNR(generated_y, image_y)
            batch_psnr.append(single_psnr)
        total_psnr.append(batch_psnr)
    total_psnr = np.array(total_psnr).flatten()
    total_psnr_avg = np.mean(total_psnr)
    total_psnr_std = np.std(total_psnr)
    print(f'avg psnr(dB):{total_psnr_avg}+/-{total_psnr_std}')
    return total_psnr, total_psnr_avg, total_psnr_std

# return median, Q1, and Q3
    
def median_percentile(y_total):
    median=np.median(y_total, axis=0)
    Q1=np.percentile(y_total, 25, axis=0)
    Q3=np.percentile(y_total, 75, axis=0)
    return Q1, median, Q3
        
# print common list after compare dsc_list and dce_list

def comm_list(dsc_path, dce_path):
    dsc_list=os.listdir(dsc_path)
    dce_list=os.listdir(dce_path)
    comm_list = [comm for comm in dsc_list if comm in dce_list]
    print(comm_list, len(comm_list))

#dsc_path="./dsc_dicom"
#dce_path="./dce_dicom"
#comm_list(dsc_path, dce_path)

#%% plot function for AIF curves

# smooth y_total using spline
    
def spline_total(y_total, perf):
    # for smoothing graph 
    time_dsc = np.linspace(0, 90, 60, endpoint=False)
    time_dce = np.linspace(0, 90, 23, endpoint=False) 
    
    # making graph smoother using spline
    time_dsc_smooth = np.linspace(time_dsc.min(), time_dsc.max(), 200)
    time_dce_smooth = np.linspace(time_dce.min(), time_dce.max(), 200)
    
    # smooth graphs
    y_total_smooth=[]
    for y in y_total:
        if perf == "DSC":
            y_smooth= spline(time_dsc, y , time_dsc_smooth)
        elif perf == "DCE":
            y_smooth= spline(time_dce, y , time_dce_smooth)
        y_total_smooth.append(y_smooth) 
    y_total_smooth=np.stack(y_total_smooth)
    return y_total_smooth

# plot all the graphs together

def plot_altogether(y_total, label, perf):
    
    # for smoothing graph 
    time_dsc = np.linspace(0, 90, 60, endpoint=False)
    time_dce = np.linspace(0, 90, 23, endpoint=False) 
    
    # making graph smoother using spline
    time_dsc_smooth = np.linspace(time_dsc.min(), time_dsc.max(), 200)
    time_dce_smooth = np.linspace(time_dce.min(), time_dce.max(), 200)
    
    # smooth graphs
    y_total_smooth=spline_total(y_total, perf)    
    
    # plot parameters
    linewidth=1
    #linestyle=':' #linestyle=linestyle
    figsize=(10,5)
    color_list=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    seed=7
    np.random.seed(seed)
    
    # plotting together    
    fig=plt.figure(figsize=figsize,constrained_layout=True)

    for idx, y in enumerate(y_total_smooth):
        rand_color=np.random.choice(color_list)
        if perf == "DSC":
            plt.plot(time_dsc_smooth, y, color=rand_color, linewidth=linewidth)
        elif perf == "DCE":
            plt.plot(time_dce_smooth, y, color=rand_color, linewidth=linewidth)

    plt.xticks(np.arange(0,91,10))
    plt.title(label)
    plt.show()
    
    filename=str(label)+"_altogether.tif"
    fig.savefig(filename, format='tif', dpi=300)
    