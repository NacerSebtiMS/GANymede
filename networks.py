# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:05:31 2019

@author: sebn3001
"""
# ========== Libraries ==========
import tensorflow as tf


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Lambda
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
# =============================

# ========== Global Variables ==========
import global_variables_wavegan as gvw
# ======================================


def apply_phaseshuffle(x, rad, pad_type='reflect'):
    # Source : https://github.com/chrisdonahue/wavegan/blob/master/wavegan.py
    
    b, x_len, nch = x.get_shape().as_list()
        
    phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)
        
    x = x[:, phase_start:phase_start+x_len]
    x.set_shape([b, x_len, nch])
    return x

# ========== WAVEGAN ==========

	# WaveGAN (https://arxiv.org/abs/1802.04208)
	# Raw audio syntesis

def build_WaveGAN_generator():
    #gvw.LATENT_DIM,gvw.D,gvw.STRIDES,gvw.KERNEL,gvw.C = args
	# Generator structure
    """
	_____________________________________________________________
	Operation 			Kernel Size 	Output Shape
	_____________________________________________________________
	Input z ∼ Uniform(−1; 1) 			(n, 100)

	Dense 1 			(100, 256d) 	(n, 256d)
	Reshape 					(n, 16, 16d)
	ReLU 						(n, 16, 16d)

	Trans Conv1D (Stride=4) 	(25, 16d, 8d) 	(n, 64, 8d)
	ReLU 						(n, 64, 8d)

	Trans Conv1D (Stride=4) 	(25, 8d, 4d) 	(n, 256, 4d)
	ReLU 						(n, 256, 4d)

	Trans Conv1D (Stride=4) 	(25, 4d, 2d) 	(n, 1024, 2d)
	ReLU 						(n, 1024, 2d)

	Trans Conv1D (Stride=4) 	(25, 2d, d) 	(n, 4096, d)
	ReLU 						(n, 4096, d)

	Trans Conv1D (Stride=4) 	(25, d, c) 	(n, 16384, c)
	Tanh 						(n, 16384, c)
	_____________________________________________________________
    """
    model = Sequential()
    model.add(Dense(256*gvw.D,input_dim=gvw.LATENT_DIM))
    model.add(Reshape((16,16*gvw.D)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(UpSampling1D(gvw.STRIDES))
    model.add(Conv1D(8*gvw.D, kernel_size=gvw.KERNEL, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(UpSampling1D(gvw.STRIDES))
    model.add(Conv1D(4*gvw.D, kernel_size=gvw.KERNEL, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(UpSampling1D(gvw.STRIDES))
    model.add(Conv1D(2*gvw.D, kernel_size=gvw.KERNEL, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(UpSampling1D(gvw.STRIDES))
    model.add(Conv1D(gvw.D, kernel_size=gvw.KERNEL, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(UpSampling1D(gvw.STRIDES))
    model.add(Conv1D(gvw.C, kernel_size=gvw.KERNEL, padding="same"))
    model.add(Activation("tanh"))
    
    if gvw.SHOW_SUMMURY : model.summary()
    
    noise = Input(shape=(gvw.LATENT_DIM,))
    output = model(noise)
    
    return Model(noise, output)

def build_WaveGAN_discriminator():
    #gvw.PHASESHUFFLE_RAD,gvw.KERNEL,gvw.STRIDES,gvw.D,gvw.C,gvw.ALPHA_LRELU = args
    #256*gvw.D, 16*gvw.D, 8*gvw.D, 4*gvw.D, 2*gvw.D = 256*gvw.D,16*gvw.D,8*gvw.D,4*gvw.D,2*gvw.D
	# Discriminator structure
    """
	_____________________________________________________________________
	Operation 				Kernel Size 	Output Shape
	_____________________________________________________________________
	Input x or G(z) 					(n, 16384, c)

	Conv1D (Stride=4) 			(25, c, d) 	(n, 4096, d)
	LReLU (α = 0.2) 					(n, 4096, d)
	Phase Shuffle (n = 2) 					(n, 4096, d)

	Conv1D (Stride=4) 			(25, d, 2d) 	(n, 1024, 2d)
	LReLU (α = 0.2) 					(n, 1024, 2d)
	Phase Shuffle (n = 2) 					(n, 1024, 2d)

	Conv1D (Stride=4) 			(25, 2d, 4d) 	(n, 256, 4d)
	LReLU (α = 0.2) 					(n, 256, 4d)
	Phase Shuffle (n = 2) 					(n, 256, 4d)

	Conv1D (Stride=4) 			(25, 4d, 8d) 	(n, 64, 8d)
	LReLU (α = 0.2) 					(n, 64, 8d)
	Phase Shuffle (n = 2) 					(n, 64, 8d)

	Conv1D (Stride=4) 			(25, 8d, 16d) 	(n, 16, 16d)
	LReLU (α = 0.2) 					(n, 16, 16d)
	Reshape 						(n, 256d)
	Dense 					(256d, 1) 	(n, 1)
	_____________________________________________________________________
    """
    phaseshuffle = lambda x : apply_phaseshuffle(x,gvw.PHASESHUFFLE_RAD)
    
    model = Sequential()
    
    model.add(Conv1D(gvw.D, kernel_size=gvw.KERNEL, strides=gvw.STRIDES, input_shape=(256*gvw.D,gvw.C), padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=gvw.ALPHA_LRELU))
    model.add(Lambda(phaseshuffle))
    
    model.add(Conv1D(2*gvw.D, kernel_size=gvw.KERNEL, strides=gvw.STRIDES, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=gvw.ALPHA_LRELU))
    model.add(Lambda(phaseshuffle))
    
    model.add(Conv1D(4*gvw.D, kernel_size=gvw.KERNEL, strides=gvw.STRIDES, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=gvw.ALPHA_LRELU))
    model.add(Lambda(phaseshuffle))
    
    model.add(Conv1D(8*gvw.D, kernel_size=gvw.KERNEL, strides=gvw.STRIDES, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=gvw.ALPHA_LRELU))
    model.add(Lambda(phaseshuffle))
    
    model.add(Conv1D(16*gvw.D, kernel_size=gvw.KERNEL, strides=gvw.STRIDES, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=gvw.ALPHA_LRELU))
    model.add(Flatten())
    #model.add(Dense(1,activation='sigmoid'))
    model.add(Dense(1))
    
    if gvw.SHOW_SUMMURY : model.summary()

    inp = Input(shape=(256*gvw.D,gvw.C,))
    validity = model(inp)
    
    return Model(inp,validity)


# =============================

# ========== GANSynth ==========

	# GANSynth (https://arxiv.org/abs/1902.08710)
"""
def build_GANSynth_generator():
	# Generator structure
	return generator

def build_GANSynth_discriminator():
	# Discriminator structure
	return discriminator

"""

# ==============================
