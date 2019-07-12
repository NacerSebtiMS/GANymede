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

def build_WaveGAN_generator(args):
    LATENT_DIM_WG,D_WG,STRIDES_WG,KERNEL_WG,C_WG = args
    D_WG256, D_WG16, D_WG8, D_WG4, D_WG2 = 256*D_WG,16*D_WG,8*D_WG,4*D_WG,2*D_WG
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
    model.add(Dense(D_WG256,input_dim=LATENT_DIM_WG))
    model.add(Reshape((16,D_WG16)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(UpSampling1D(STRIDES_WG))
    model.add(Conv1D(D_WG8, kernel_size=KERNEL_WG, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(UpSampling1D(STRIDES_WG))
    model.add(Conv1D(D_WG4, kernel_size=KERNEL_WG, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(UpSampling1D(STRIDES_WG))
    model.add(Conv1D(D_WG2, kernel_size=KERNEL_WG, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(UpSampling1D(STRIDES_WG))
    model.add(Conv1D(D_WG, kernel_size=KERNEL_WG, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(UpSampling1D(STRIDES_WG))
    model.add(Conv1D(C_WG, kernel_size=KERNEL_WG, padding="same"))
    model.add(Activation("tanh"))
    
    model.summary()
    
    noise = Input(shape=(LATENT_DIM_WG,))
    output = model(noise)
    
    return Model(noise, output)

def build_WaveGAN_discriminator(args):
    PHASESHUFFLE_RAD,KERNEL_WG,STRIDES_WG,D_WG,C_WG,ALPHA_WG = args
    D_WG256, D_WG16, D_WG8, D_WG4, D_WG2 = 256*D_WG,16*D_WG,8*D_WG,4*D_WG,2*D_WG
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
    phaseshuffle = lambda x : apply_phaseshuffle(x,PHASESHUFFLE_RAD)
    
    model = Sequential()
    
    model.add(Conv1D(D_WG, kernel_size=KERNEL_WG, strides=STRIDES_WG, input_shape=(D_WG256,C_WG), padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=ALPHA_WG))
    model.add(Lambda(phaseshuffle))
    
    model.add(Conv1D(D_WG2, kernel_size=KERNEL_WG, strides=STRIDES_WG, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=ALPHA_WG))
    model.add(Lambda(phaseshuffle))
    
    model.add(Conv1D(D_WG4, kernel_size=KERNEL_WG, strides=STRIDES_WG, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=ALPHA_WG))
    model.add(Lambda(phaseshuffle))
    
    model.add(Conv1D(D_WG8, kernel_size=KERNEL_WG, strides=STRIDES_WG, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=ALPHA_WG))
    model.add(Lambda(phaseshuffle))
    
    model.add(Conv1D(D_WG16, kernel_size=KERNEL_WG, strides=STRIDES_WG, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=ALPHA_WG))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    
    model.summary()

    inp = Input(shape=(D_WG256,C_WG,))
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
