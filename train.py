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
from networks import build_WaveGAN_generator, build_WaveGAN_discriminator
from loader import decode_extract_and_batch
#from gp_loss import partial_gp_loss
# ===========================

# ========== Global Variables ==========
import global_variables_wavegan as gvw
# ======================================

# ========== WaveGAN ==========
def train_wavegan():
    # Load dataset
    x = decode_extract_and_batch(
            fps=gvw.FPS,
            batch_size=gvw.BATCH_SIZE,
            slice_len=gvw.SLICE_LEN,
            decode_fs=gvw.DECODE_FS,
            decode_num_channels=gvw.DECODE_NUM_CHANNELS,
            decode_fast_wav=gvw.DECODE_FAST_WAV)
    
    
    
    # Make z vector
    z = tf.random_uniform([gvw.BATCH_SIZE, gvw.LATENT_DIM], -1., 1., dtype=tf.float32)
    
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
    
    """
    # Create the optimizer
    optimizer = Adam(ALPHA_A_WG,beta_1=BETA1_A_WG,beta_2=BETA2_A_WG)
    
    # Build the loss function
    gradient_penalty_loss = partial_gp_loss(GP_INPUT_WG)
    
    # Build and compile the discriminator    
    discriminator = net.build_WaveGAN_discriminator(DISCRIMINATOR_INPUT_WG)
    discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrcis=['accuracy'])
    
    # Build generator
    generator = net.build_WaveGAN_generator(GENERATOR_INPUT_WG)
    
    # Make z vector and input it to the generator
    #z = tf.random_uniform([BATCH_SIZE, LATENT_DIM_WG], -1., 1., dtype=tf.float32)
    z = Input(shape=(LATENT_DIM_WG,))
    output = generator(z)
    
    # For the combined model we will only train the generator
    discriminator.trainable = False
    
    # The discriminator takes generated images as input and determines validity
    valid = discriminator(output)
    
    # The combined model  (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    """
    
    
 
    
# ______________________________
    # Saving repository
    
    # Discriminator training
    
    # Generator Training
    
# =============================