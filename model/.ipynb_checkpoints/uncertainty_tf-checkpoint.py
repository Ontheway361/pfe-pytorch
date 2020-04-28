#!/usr/bin/env python3
#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from IPython import embed

batch_norm_params = {
    'decay': 0.995,
    'epsilon': 0.001,
    'center': True,
    'scale': True,
    'updates_collections': None,
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}

batch_norm_params_sigma = {
    'decay': 0.995,
    'epsilon': 0.001,
    'center': False,
    'scale': False,
    'updates_collections': None,
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],}

def scale_and_shift(x, gamma_init=1.0, beta_init=0.0):
    num_channels = x.shape[-1].value
    with tf.variable_scope('scale_and_shift'):
        gamma = tf.get_variable('alpha', (),
                        initializer=tf.constant_initializer(gamma_init),
                        regularizer=slim.l2_regularizer(0.0),
                        dtype=tf.float32)
        beta = tf.get_variable('gamma', (),
                        initializer=tf.constant_initializer(beta_init),
                        dtype=tf.float32)
        x = gamma * x +  beta

        return x   
    

def inference(inputs, embedding_size, phase_train, 
        weight_decay=5e-4, reuse=None, scope='UncertaintyModule'):
    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=tf.nn.relu):
        with tf.variable_scope(scope, [inputs], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):
                print('UncertaintyModule input shape:', [dim.value for dim in inputs.shape])

                net = slim.flatten(inputs)
                
                embed()
                net = slim.fully_connected(net, embedding_size, scope='fc1',
                    normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, 
                    activation_fn=tf.nn.relu)


                log_sigma_sq = slim.fully_connected(net, embedding_size, scope='fc_log_sigma_sq',
                    normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params_sigma,
                    activation_fn=None)
          
                # Share the gamma and beta for all dimensions
                log_sigma_sq = scale_and_shift(log_sigma_sq, 1e-4, -7.0)

                # Add epsilon for sigma_sq for numerical stableness                
                log_sigma_sq = tf.log(1e-6 + tf.exp(log_sigma_sq))

    return log_sigma_sq



if __name__ == "__main__":

    mu_data = np.array([[-1.7847768 , -1.0991699 ,  1.4248079 ],
                        [ 1.0405252 ,  0.35788524,  0.7338794 ],
                        [ 1.0620259 ,  2.1341069 , -1.0100055 ],
                        [-0.00963581,  0.39570177, -1.5577421 ],
                        [-1.064951  , -1.1261107 , -1.4181522 ],
                        [ 1.008275  , -0.84791195,  0.3006532 ],
                        [ 0.31099692, -0.32650718, -0.60247767]], dtype=np.float64)
    
    muX = tf.convert_to_tensor(mu_data)
    log_sigma_sq = inference(muX, 3, True)
    print(diff)