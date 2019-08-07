# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:28:15 2019

@author: sebn3001
"""
# ========== Libraries ==========
import numpy as np

import tensorflow as tf

from keras import backend as K
from keras.layers import Input
from keras.layers.merge import _Merge
from functools import partial
# =============================

# ========== Global Variables ==========
import global_variables_wavegan as gvw
# ======================================


# ========== Loss function ==========
class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((gvw.BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
    
def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    print(type(grads))
    print(type(var_list))
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in tf.map_fn(zip,(var_list, grads))]

def gradient_penalty_loss(y_true, y_pred, averaged_samples,gradient_penalty_weight):
    # Source : https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py 
    
	# first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    
    #gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients = _compute_gradients(y_pred, averaged_samples)[0]
    
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

def partial_gp_loss(y_true,y_pred):
    real_samples = Input(shape=gvw.NOISE_SIZE)
    generated_samples = Input(shape=(gvw.GP_SIZE2,))
    averaged_samples = RandomWeightedAverage()([real_samples,generated_samples])
    gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples,gradient_penalty_weight=gvw.GRADIENT_PENALTY_WEIGHT)
    return gp_loss(y_true,y_pred)
# ===================================