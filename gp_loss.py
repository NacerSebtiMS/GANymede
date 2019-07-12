# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:28:15 2019

@author: sebn3001
"""
# ========== Libraries ==========
import numpy as np

from keras import backend as K
# =============================


# ========== Loss function ==========

def gradient_penalty_loss(y_true, y_pred, averaged_samples,gradient_penalty_weight):
    # Source : https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py 
    
	# first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)
# ===================================