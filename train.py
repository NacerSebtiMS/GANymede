# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:34:55 2019

@author: sebn3001
"""

# ========== Libraries ==========
import tensorflow as tf

#import numpy as np

#import matplotlib.pyplot as plt

#import sys,os,shutil

#from keras.models import Sequential, Model#, load_model
#from keras.layers import Input, Dense, Reshape, Flatten, Lambda
#from keras.layers import BatchNormalization, Activation
#from keras.layers.advanced_activations import LeakyReLU
#from keras.layers.convolutional import UpSampling1D, Conv1D
#from keras.optimizers import Adam
# =============================

# ========== Files ==========
import networks as net
import loader
#import gp_loss as gp
# ===========================

# ========== WaveGAN ==========
def train_wavegan(args):
    epochs, batch_size, save_interval, fps, slice_len, decode_fs, decode_num_channels, decode_fast_wav, latent_dim_wg = args
    # Load dataset
    x = loader.decode_extract_and_batch(
            fps=fps,
            batch_size=batch_size,
            slice_len=slice_len,
            decode_fs=decode_fs,
            decode_num_channels=decode_num_channels,
            decode_fast_wav=decode_fast_wav)
    
    
    
    # Make z vector
    z = tf.random_uniform([args.train_batch_size, args.wavegan_latent_dim], -1., 1., dtype=tf.float32)
    
    # Make generator
    
    # Print G summary
    
    # Summarize
    
    # Make real discriminator
    
    # Print D summary
    
    # Make fake discriminator
    
    # Create loss
    
    # Create (recommended) optimizer
    
    # Create training ops
    
    # Run training
    
    
 
    
# ______________________________
    # Saving repository
    
    # Discriminator training
    
    # Generator Training
    
# =============================