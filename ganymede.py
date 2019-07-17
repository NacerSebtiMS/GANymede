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
# =============================

# ========== Files ==========
from train import train_wavegan
# ===========================

# ========== Activators ==========

	# Train models
TRAIN = False

	# Predict using the last generator
#PREDICT = True

	# Predict using the same noise
#PREDICT_SAME_NOISE = True

# ================================

# ========== Main ==========
if __name__ == '__main__' :
    if TRAIN :
        train_wavegan()
    
# ==========================