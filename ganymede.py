# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 20:31:47 2019

@author: sebn3001
"""

# ========== Libraries ==========
#import tensorflow as tf

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
#import gp_loss as gp
#import train
# ===========================

# ========== Activators ==========

	# Train models
TRAIN = False

	# Predict using the last generator
PREDICT = True

	# Predict using the same noise
PREDICT_SAME_NOISE = True

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

		# Adam Optimizer Parameters
ALPHA_A_WG = 1e-4
BETA1_A_WG = 0.5
BETA2_A_WG = 0.9
	# =========================

	# Saving Paths
VERSION = '1.0'
MODEL_PATH = './models'
TRAIN_GENERATION_PATH = './generated/training'
PREDICT_PATH = './generated/predict'

# ===============================

# ========== Packing Vars ==========

# WaveGAN
GENERATOR_INPUT_WG = [LATENT_DIM_WG,D_WG,STRIDES_WG,KERNEL_WG,C_WG]
DISCRIMINATOR_INPUT_WG = [PHASESHUFFLE_RAD,KERNEL_WG,STRIDES_WG,D_WG,C_WG,ALPHA_WG]
TRAIN_INPUT_WG = [EPOCH,SAVE_INTERVAL,BATCH_SIZE]
# ================================== 

# ========== Main ==========
    
generator = net.build_WaveGAN_generator(GENERATOR_INPUT_WG)
discriminator = net.build_WaveGAN_discriminator(DISCRIMINATOR_INPUT_WG)

# ==========================