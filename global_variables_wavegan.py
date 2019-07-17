# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 22:32:11 2019

@author: sebn3001
"""
import tensorflow as tf

# ========== Variables ==========

	# Tuning training
EPOCH = 700
SAVE_INTERVAL = 50
BATCH_SIZE = 32

	# === WaveGAN Variables ===
N = 64 				# Batch size
D = 64 				# Model size
C = 1				# Num channels
ALPHA_LRELU = 0.2	# LReLu parameter
STRIDES = 4			# Stride for Trans Conv in the generator
LATENT_DIM = 100	# Size of the noise
KERNEL = 25			# Kernel size

PHASESHUFFLE_RAD = 2    # Phase shuffle that randomize phase of each channel

NOISE = tf.random_uniform([BATCH_SIZE, LATENT_DIM], -1., 1., dtype=tf.float32)

        # Decode input parameters
        
SLICE_LEN = 8192 			# Length of the sliceuences in samples or feature timesteps
DECODE_FS = 0 				# (Re-)sample rate for decoded audio files
DECODE_NUM_CHANNELS = 1 	# Number of channels for decoded audio files
DECODE_FAST_WAV = True 		# Using Spicy instead of Librosa for faster training

		# Adam Optimizer Parameters
ALPHA_ADAM = 1e-4
BETA1_ADAM = 0.5
BETA2_ADAM = 0.9

        # GP Loss parameters
        
NOISE_SIZE = NOISE.shape[1:]
GP_SIZE2 = 100
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
#GENERATOR_INPUT_WG = [LATENT_DIM_WG,D_WG,STRIDES_WG,KERNEL_WG,C_WG]
#DISCRIMINATOR_INPUT_WG = [PHASESHUFFLE_RAD,KERNEL_WG,STRIDES_WG,D_WG,C_WG,ALPHA_WG]
#TRAIN_INPUT_WG = [EPOCH,SAVE_INTERVAL,BATCH_SIZE, FPS, SLICE_LEN, DECODE_FS, DECODE_NUM_CHANNELS, DECODE_FAST_WAV, LATENT_DIM_WG]
#GP_INPUT_WG = [NOISE_SIZE,SIZE2,GRADIENT_PENALTY_WEIGHT]
# ================================== 