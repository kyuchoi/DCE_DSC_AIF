# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:15:31 2019

@author: CNDLMembers
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from random import gauss
from utils import *

seed=11342513
np.random.seed(seed)

### load train and test data: (None,60)
x_train=np.load("x_train.npy")
y_train=np.load("y_train.npy")
x_test=np.load("x_test.npy")
y_test=np.load("y_test.npy")

### resample y (AIF_DSC) to match x (AIF_DCE) for the time-scale: i.e. temporal normalization
choose=0 # choose interpolation methods: 1=cubic spline vs 0=polynomial, which makes no significant difference 
plot=0 # choose whether to display the plots

y_train_resampled=resample_y(y_train, choose, plot) # (520,60) -> (520,60) but 0-240
y_test_resampled=resample_y(y_test, choose, plot) # (252,60) -> (252,60) but 0-240

### add time-stpes in the end to make 2D STFT matrix size
add_steps=3 # only checked COLA when 3
x_train=x_add_steps(x_train, add_steps) # (520, 60) -> (520,63)
x_test=x_add_steps(x_test, add_steps) # (252, 60) -> (252,63)
y_train=x_add_steps(y_train_resampled, add_steps) # (520, 60) -> (520,63)
y_test=x_add_steps(y_test_resampled, add_steps) # (252, 60) -> (252,63)

#%%
nperseg=14
noverlap=7 # noverlap is half of nperseg by default

# initialize
Zxx_train=[]
Zyy_train=[]

Zxx_n_train=[]
Zyy_n_train=[]

for idx in range(x_train.shape[0]):
    x=x_train[idx].squeeze()
    y=y_train[idx].squeeze()
    
    x_lr=np.flip(x, axis=0)

    f_x,t_x,Zxx=signal.stft(x,nperseg=nperseg, noverlap=noverlap)
    f_x_lr,t_x_lr,Zxx_lr=signal.stft(x_lr,nperseg=nperseg, noverlap=noverlap)
            
    Zxx_train.append(Zxx)
    Zxx_train.append(Zxx_lr)
       
    # augment using flipping (horizontal only)
    y_lr=np.flip(y, axis=0)
    
    f_y,t_y,Zyy=signal.stft(y,nperseg=nperseg, noverlap=noverlap)
    f_y_lr,t_y_lr,Zyy_lr=signal.stft(y_lr,nperseg=nperseg, noverlap=noverlap)
    
    Zyy_train.append(Zyy)
    Zyy_train.append(Zyy_lr)
    
    # add noise to augment
    noise=[gauss(0.0, 0.2) for i in range(x_train.shape[1])] 
    x_n=x+noise
    y_n=y+noise
    
    x_n_lr=np.flip(x_n, axis=0)
    
    f_x_n,t_x_n,Zxx_n=signal.stft(x_n,nperseg=nperseg, noverlap=noverlap)
    f_x_n_lr,t_x_n_lr,Zxx_n_lr=signal.stft(x_n_lr,nperseg=nperseg, noverlap=noverlap)    
    
    Zxx_n_train.append(Zxx_n)
    Zxx_n_train.append(Zxx_n_lr)
    
    y_n_lr=np.flip(y_n, axis=0)
    
    f_y_n,t_y_n,Zyy_n=signal.stft(y_n,nperseg=nperseg, noverlap=noverlap)
    f_y_n_lr,t_y_n_lr,Zyy_n_lr=signal.stft(y_n_lr,nperseg=nperseg, noverlap=noverlap)

    Zyy_n_train.append(Zyy_n)
    Zyy_n_train.append(Zyy_n_lr)
    
Zxx_train=np.stack(Zxx_train)
Zyy_train=np.stack(Zyy_train)
Zxx_n_train=np.stack(Zxx_n_train)
Zyy_n_train=np.stack(Zyy_n_train)

print(Zxx_train.shape, Zxx_n_train.shape)
print(Zyy_train.shape, Zyy_n_train.shape)

Zxx_train_aug=np.vstack((Zxx_train, Zxx_n_train))
Zyy_train_aug=np.vstack((Zyy_train, Zyy_n_train))

#%% make complex number into 2 channels
# for x_train
print(Zxx_train_aug[0].real.shape, Zxx_train_aug[0].imag.shape)
x_train_fin_tot=[]
for idx in range(Zxx_train_aug.shape[0]):
    x_train_fin=np.concatenate((np.expand_dims(Zxx_train_aug[idx].real, axis=-1), np.expand_dims(Zxx_train_aug[idx].imag, axis=-1)), axis=-1)
    x_train_fin_tot.append(x_train_fin)    
Zxx_train_aug_chan=np.stack(x_train_fin_tot)    

# for y_train
print(Zyy_train_aug[0].real.shape, Zyy_train_aug[0].imag.shape)
y_train_fin_tot=[]
for idx in range(Zyy_train_aug.shape[0]):
    y_train_fin=np.concatenate((np.expand_dims(Zyy_train_aug[idx].real, axis=-1), np.expand_dims(Zyy_train_aug[idx].imag, axis=-1)), axis=-1)
    y_train_fin_tot.append(y_train_fin)    
Zyy_train_aug_chan=np.stack(y_train_fin_tot)    

#%% stft test set as well

Zxx_test_tot=[]
Zyy_test_tot=[]

for idx in range(x_test.shape[0]):
    x=x_test[idx].squeeze()
    y=y_test[idx].squeeze()
    
    f_x,t_x,Zxx_test=signal.stft(x,nperseg=nperseg)
    f_y,t_y,Zyy_test=signal.stft(y,nperseg=nperseg)
    
    Zxx_test_tot.append(Zxx_test)
    Zyy_test_tot.append(Zyy_test)
Zxx_test_tot=np.stack(Zxx_test_tot)
Zyy_test_tot=np.stack(Zyy_test_tot)

# make real and imaginary parts as two channels
x_test_fin_tot=[]
for idx in range(Zxx_test_tot.shape[0]):
    x_test_fin=np.concatenate((np.expand_dims(Zxx_test_tot[idx].real, axis=-1), np.expand_dims(Zxx_test_tot[idx].imag, axis=-1)), axis=-1)
    x_test_fin_tot.append(x_test_fin)    
Zxx_test_aug_chan=np.stack(x_test_fin_tot)    

y_test_fin_tot=[]
for idx in range(Zyy_test_tot.shape[0]):
    y_test_fin=np.concatenate((np.expand_dims(Zyy_test_tot[idx].real, axis=-1), np.expand_dims(Zyy_test_tot[idx].imag, axis=-1)), axis=-1)
    y_test_fin_tot.append(y_test_fin)    
Zyy_test_aug_chan=np.stack(y_test_fin_tot)    

#%% save as npy
if choose == 1:
    np.save("Zxx_train.npy", Zxx_train_aug_chan) 
    np.save("Zyy_train_spl.npy", Zyy_train_aug_chan)
    np.save("Zxx_test.npy", Zxx_test_aug_chan)
    np.save("Zyy_test_spl.npy", Zyy_test_aug_chan)
else:
    np.save("Zxx_train.npy", Zxx_train_aug_chan) 
    np.save("Zyy_train_poly.npy", Zyy_train_aug_chan)
    np.save("Zxx_test.npy", Zxx_test_aug_chan)
    np.save("Zyy_test_poly.npy", Zyy_test_aug_chan)
#%% recon example from real and imaginary part

sample_size=10
fig, ax = plt.subplots(sample_size, figsize=(5,60))
for idx in range(sample_size):
    t_rec_complex,Zxx_rec_complex=signal.istft(Zxx_train_aug_chan[idx,:,:,0]+1j*Zxx_train_aug_chan[idx,:,:,1], nperseg=nperseg, noverlap=noverlap)
    ax[idx].plot(t_rec_complex, Zxx_rec_complex, label='DCE')
    t_rec_complex,Zyy_rec_complex=signal.istft(Zyy_train_aug_chan[idx,:,:,0]+1j*Zyy_train_aug_chan[idx,:,:,1], nperseg=nperseg, noverlap=noverlap)
    ax[idx].plot(t_rec_complex, Zyy_rec_complex, label='DSC')
    ax[idx].legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()