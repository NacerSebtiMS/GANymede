# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 20:31:47 2019

@author: sebn3001
"""

# ========== Libraries ==========
import tensorflow as tf

#import numpy as np

#import matplotlib.pyplot as plt

#import sys,os,shutil

from keras.models import Model#, load_model
from keras.layers import Input#, Dense, Reshape, Flatten, Lambda
#from keras.layers import BatchNormalization, Activation
#from keras.layers.advanced_activations import LeakyReLU
#from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.optimizers import Adam
# =============================

# ========== Files ==========
import networks as net
from gp_loss import partial_gp_loss
import train
# ===========================

# ========== Activators ==========

	# Train models
TRAIN = False

	# Predict using the last generator
#PREDICT = True

	# Predict using the same noise
#PREDICT_SAME_NOISE = True

# ================================

# ========== Variables ==========

	# Tuning training
EPOCH = 700
SAVE_INTERVAL = 50
BATCH_SIZE = 32

	# === WaveGAN Variables ===
N_WG = 64 			# Batch size
D_WG = 64 			# Model size
D_WG256, D_WG16, D_WG8, D_WG4, D_WG2 = 256*D_WG,16*D_WG,8*D_WG,4*D_WG,2*D_WG
C_WG = 1			# Num channels
ALPHA_WG = 0.2		# LReLu parameter
STRIDES_WG = 4		# Stride for Trans Conv in the generator
LATENT_DIM_WG = 100	# Size of the noise
KERNEL_WG = 25		# Kernel size

PHASESHUFFLE_RAD = 2    # Phase shuffle that randomize phase of each channel

NOISE = tf.random_uniform([BATCH_SIZE, LATENT_DIM_WG], -1., 1., dtype=tf.float32)

        # Decode input parameters
        
SLICE_LEN = 8192 # Length of the sliceuences in samples or feature timesteps
DECODE_FS = 0 # (Re-)sample rate for decoded audio files
DECODE_NUM_CHANNELS = 1 # Number of channels for decoded audio files
DECODE_FAST_WAV = True # Using Spicy instead of Librosa for faster training

		# Adam Optimizer Parameters
ALPHA_A_WG = 1e-4
BETA1_A_WG = 0.5
BETA2_A_WG = 0.9

        # GP Loss parameters
        
NOISE_SIZE = NOISE.shape[1:]
SIZE2 = 100
GRADIENT_PENALTY_WEIGHT = 10

	# =========================

	# Saving Paths
VERSION = '1.0'
MODEL_PATH = './models'
DATASET_PATH_DRUMS = './dataset/drums/train'
DATASET_PATH_PIANO = './dataset/piano/train'
DATASET_PATH_SC09 = './dataset/sc09/train'
FPS = [DATASET_PATH_DRUMS, DATASET_PATH_PIANO, DATASET_PATH_SC09]
#TRAIN_GENERATION_PATH = './generated/training'
#PREDICT_PATH = './generated/predict'

# ===============================

# ========== Packing Vars ==========

# WaveGAN
GENERATOR_INPUT_WG = [LATENT_DIM_WG,D_WG,STRIDES_WG,KERNEL_WG,C_WG]
DISCRIMINATOR_INPUT_WG = [PHASESHUFFLE_RAD,KERNEL_WG,STRIDES_WG,D_WG,C_WG,ALPHA_WG]
TRAIN_INPUT_WG = [EPOCH,SAVE_INTERVAL,BATCH_SIZE, FPS, SLICE_LEN, DECODE_FS, DECODE_NUM_CHANNELS, DECODE_FAST_WAV, LATENT_DIM_WG]
GP_INPUT_WG = [NOISE_SIZE,SIZE2,GRADIENT_PENALTY_WEIGHT]
# ================================== 

# ========== Main ==========
if __name__ == '__main__' : 
    
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
    
    
    if TRAIN :
        train.train_wavegan(TRAIN_INPUT_WG)
    
# ==========================