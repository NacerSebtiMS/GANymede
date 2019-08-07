# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:34:55 2019

@author: sebn3001
"""

# ========== Libraries ==========
import tensorflow as tf

import numpy as np

from keras.models import Model, load_model
from keras.layers import Input
from keras.optimizers import Adam

from scipy.io import wavfile

import os,shutil

import datetime
# =============================

# ========== Files ==========
from networks import build_WaveGAN_generator, build_WaveGAN_discriminator
#from loader import decode_extract_and_batch
from load import load_dataset, batch
from gp_loss import partial_gp_loss
# ===========================

# ========== Global Variables ==========
import global_variables_wavegan as gvw
# ======================================

def save_audio(audio,path):
    audio = audio * (pow(2,16)/2)
    wavfile.write(path,gvw.DECODE_FS,audio.astype("int16"))
    
def numb(n):
    if n < 10 : return "0"+str(n)
    else : return str(n)

def manage_file(path):
    if os.path.isdir(path): shutil.rmtree(path)
    os.makedirs(path)

# ========== WaveGAN ==========
def train_wavegan():    
    # Load dataset
    """
    data = decode_extract_and_batch(
            fps=gvw.FPS,
            batch_size=gvw.BATCH_SIZE,
            slice_len=gvw.SLICE_LEN,
            decode_fs=gvw.DECODE_FS,
            decode_num_channels=gvw.DECODE_NUM_CHANNELS,
            decode_fast_wav=gvw.DECODE_FAST_WAV)[:,:,0]
    """    
    x = load_dataset()
    # Make generator
    generator = build_WaveGAN_generator()
    print("Successfully builded generator")
    
    # Make z vector
    z = Input(shape=(gvw.LATENT_DIM,))
    output = generator(z)
    print("Output generated")
    
    # Create optimizer
    optimizer = Adam(gvw.ALPHA_ADAM,beta_1=gvw.BETA1_ADAM,beta_2=gvw.BETA2_ADAM)
    
    # Init loss
    if gvw.LOSS == "wgan-gp" :
        # Init WGAN-GP loss
        loss = partial_gp_loss
        loss.__name__ = 'gradient_penalty_loss'
    else : loss = gvw.LOSS
    
    # Make and compile discriminator
    discriminator = build_WaveGAN_discriminator()
    # For the combined model we will only train the generator
    discriminator.trainable = False
    discriminator.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
    print("Successfully builded discriminator")
    
    # The discriminator takes generated images as input and determines validity
    okay = discriminator(output)
    
    # The combined model  (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    combined = Model(z, okay)
    combined.compile(loss=loss, optimizer=optimizer)  
    print("Successfully builded combined model")
    
    okay = np.ones((gvw.BATCH_SIZE, 1))
    fake = np.zeros((gvw.BATCH_SIZE, 1))
    
    # Create directories
    manage_file(gvw.GENERATION_PATH)
    for i in range(gvw.SAVE_INTERVAL,gvw.EPOCH+1,gvw.SAVE_INTERVAL) : manage_file(gvw.GENERATION_PATH+ "EPOCH_" +str(i)+"/")
    manage_file(gvw.MODEL_PATH)
    
    # Run training
    for epoch in range(gvw.EPOCH) :
        if epoch == 1 : 
            t0 = datetime.datetime.now()
            t_temp = t0
        
        
        # ========== Discriminator training ==========
        
        # Batch audio
        audio = batch(x)
        
        # Sample noise and generate audio
        print("# =======",epoch,"======= #")
        noise = tf.random_uniform([gvw.BATCH_SIZE, gvw.LATENT_DIM], -1., 1., dtype=tf.float32).eval(session=tf.Session())
        gen_audio = generator.predict(noise)
        print("Predicted noise")
        
        # Train discriminator
        d_loss_okay = discriminator.train_on_batch(audio, okay)
        print("Successfully trained okay loss")
        d_loss_fake = discriminator.train_on_batch(gen_audio, fake)
        print("Successfully trained fake loss")
        d_loss = 0.5 * np.add(d_loss_okay, d_loss_fake)
        
        # ============================================
        
        # ========== Generator training ==========
        
        # Train the generator (wants discriminator to mistake images as real)
        g_loss = combined.train_on_batch(noise, okay)
        print("Successfully trained combined model")
        
        # ========================================
        
        # Plot the progress
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] - %.2f%%" % (epoch, d_loss[0], 100*d_loss[1], g_loss, 100*epoch/gvw.EPOCH))
        if epoch > 0 :
            
            t = datetime.datetime.now()
            remaining = (gvw.EPOCH - epoch -1) * (t-t0) / (epoch)
            h = remaining.seconds // 3600
            m = (remaining.seconds - h*3600) // 60
            s = remaining.seconds - h*3600 - m*60
            print("Epoch executed in %.2fs" % ((t-t_temp).total_seconds()) )
            print("Remaining time ",end="")
            if h > 0 : print(h,"h",sep ="",end=" ")
            if m > 0 : print(m,"m",sep ="",end=" ")
            if s > 0 : print(s,"s",sep ="",end=" ")
            print("")
            t_temp = t
        
        if (epoch+1) % gvw.SAVE_INTERVAL == 0 :
            for i in range(gen_audio.shape[0]) :
                path1 = gvw.GENERATION_PATH+  "EPOCH_" +str(epoch+1)+"/"+ str(epoch+1) + "-" + numb(i) + '.wav' 
                save_audio(gen_audio[i,:,0],path1)
            path2 = gvw.MODEL_PATH+"generator_"+str(epoch+1)+".h5"
            generator.save(path2)
            print("Successfully saved generated sample and generator model")
    
def predict_wavegan(same_noise=False,epoch=gvw.EPOCH):    
    generator = load_model(gvw.MODEL_PATH+"generator_"+str(epoch)+".h5")
    if same_noise : noise = tf.random_uniform([gvw.BATCH_SIZE, gvw.LATENT_DIM], -1., 1., dtype=tf.float32).eval(session=tf.Session())
    else : noise = gvw.NOISE
    prediction = generator.predict(noise)
    
    directory = gvw.PREDICTION_PATH+ "EPOCH_" +str(epoch)+"/"
    manage_file(directory)
    for i in range(prediction.shape[0]) :
        filename = str(epoch) + "-" + numb(i) + ".wav"
        path = directory +  filename
        save_audio(prediction[i,:,0],path)
        print("Saved %s" % (filename) )
    print("Successfully generated %s samples with epoch's %s generator" % (prediction.shape[0], epoch) )
# =============================