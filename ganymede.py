# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 20:31:47 2019

@author: sebn3001
"""

# ========== Libraries ==========
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session

import os,shutil
# =============================

# ========== Files ==========
from train import train_wavegan, predict_wavegan
import global_variables_wavegan as gvw
# ===========================

# ========== Activators ==========

	# Train models
TRAIN = True

	# Predict using the last generator
PREDICT = False

	# Predict using the same noise
#PREDICT_SAME_NOISE = True

# ================================
    
def manage_file(path):
    if os.path.isdir(path): shutil.rmtree(path)
    os.makedirs(path)

# ========== Main ==========
if __name__ == '__main__' :
    
    # Allow growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    set_session(sess)
    
    if TRAIN :
        # Create directories
        manage_file(gvw.GENERATION_PATH)    # Generated audio
        for i in range(gvw.SAVE_INTERVAL,gvw.EPOCH+1,gvw.SAVE_INTERVAL) : manage_file(gvw.GENERATION_PATH+ "EPOCH_" +str(i)+"/")
        manage_file(gvw.MODEL_PATH)         # Generator model
        
        train_wavegan()
        
    if PREDICT :
        manage_file(gvw.PREDICTION_PATH)
        predict_wavegan()
    
# ==========================