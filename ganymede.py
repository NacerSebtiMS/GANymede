# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 20:31:47 2019

@author: sebn3001
"""

# ========== Libraries ==========
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
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
PREDICT_SAME_NOISE = True

# ================================
    


# ========== Main ==========
if __name__ == '__main__' :
    
    # Allow growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    set_session(sess)
    
    if TRAIN :
        train_wavegan()
        
    if PREDICT :       
        for i in range(gvw.SAVE_INTERVAL,gvw.EPOCH+1,gvw.SAVE_INTERVAL) : predict_wavegan(PREDICT_SAME_NOISE,i)
    
# ==========================