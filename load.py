# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:44:38 2019

@author: sebn3001
"""
# ========== Libraries ==========
from scipy.io import wavfile
import numpy as np
# =============================

# ========== Global Variables ==========
import global_variables_wavegan as gvw
# ======================================

def load_dataset():
    l = list()
    for file in gvw.FPS:
        _,d = wavfile.read(file)
        d = d/(pow(2,16)/2) # Rescale -1 to 1
        
        # Add zeros to have the good data shape
        if d.shape[0] - gvw.SLICE_LEN < 0 : d = np.concatenate( (d, np.ones((gvw.SLICE_LEN-d.shape[0])) ) )
        
        d= np.expand_dims(d,axis=1)
        l.append(d)
    data=np.array(l)
    return data

def batch(dataset):
    idx = np.random.randint(0, dataset.shape[0], gvw.BATCH_SIZE)
    return dataset[idx]