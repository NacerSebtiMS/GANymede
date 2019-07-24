# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 22:32:11 2019

@author: sebn3001
"""
import tensorflow as tf
from os import listdir
from os.path import isfile, join

# ========== Functions ==========
def extract_all_files(path):
    return [path+f for f in listdir(path) if isfile(join(path, f))]
# ===============================

# ========== Variables ==========

	# Tuning training
EPOCH = 700	# 700 Epochs for the SC09 dataset and 300 for sound datasets
BATCH_SIZE = 64
SHOW_SUMMURY = False # Put to True to be able to print the keras summury of the discriminator and the generator

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
        
SLICE_LEN = 16384 			# Length of the sliceuences in samples or feature timesteps
DECODE_FS = 16000			# (Re-)sample rate for decoded audio files
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
SAVE_INTERVAL = 50
VERSION = '1.0'
MODEL_PATH = './models/' + VERSION + '/'
GENERATION_PATH = './generated/' + VERSION + '/'
PREDICTION_PATH = GENERATION_PATH + 'predicted/'

DATASET_PATH_DRUMS = './dataset/drums/train/'
DATASET_PATH_PIANO = './dataset/piano/train/'
DATASET_PATH_SC09 = './dataset/sc09/train/'

FPS = extract_all_files(DATASET_PATH_SC09)

# ===============================