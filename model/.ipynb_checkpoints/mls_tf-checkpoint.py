#!/usr/bin/env python3
#-*- coding:utf-8 -*-
"""
Created on 2020/04/23
author: lujie
"""
import numpy as np
import tensorflow as tf
from IPython import embed

def negative_MLS(X, Y, sigma_sq_X, sigma_sq_Y, mean=False):
    with tf.name_scope('negative_MLS'):
        if mean:
            D = X.shape[1].value

            Y = tf.transpose(Y)
            XX = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
            YY = tf.reduce_sum(tf.square(Y), 0, keep_dims=True)
            XY = tf.matmul(X, Y)
            diffs = XX + YY - 2*XY

            sigma_sq_Y = tf.transpose(sigma_sq_Y)
            sigma_sq_X = tf.reduce_mean(sigma_sq_X, axis=1, keep_dims=True)
            sigma_sq_Y = tf.reduce_mean(sigma_sq_Y, axis=0, keep_dims=True)
            sigma_sq_fuse = sigma_sq_X + sigma_sq_Y

            diffs = diffs / (1e-8 + sigma_sq_fuse) + D * tf.log(sigma_sq_fuse)

            return diffs
        else:
            # D = X.shape[1].value
            D = X.shape[1]
            X = tf.reshape(X, [-1, 1, D])
            Y = tf.reshape(Y, [1, -1, D])
            sigma_sq_X = tf.reshape(sigma_sq_X, [-1, 1, D])
            sigma_sq_Y = tf.reshape(sigma_sq_Y, [1, -1, D])
            sigma_sq_fuse = sigma_sq_X + sigma_sq_Y
            diffs = tf.square(X-Y) / (1e-10 + sigma_sq_fuse) + tf.math.log(sigma_sq_fuse)
            return tf.reduce_sum(diffs, axis=2)

def mutual_likelihood_score_loss(labels, mu, log_sigma_sq):

    with tf.name_scope('MLS_Loss'):

        batch_size = tf.shape(mu)[0]
        diag_mask = tf.eye(batch_size, dtype=tf.bool)
        non_diag_mask = tf.logical_not(diag_mask)

        sigma_sq = tf.exp(log_sigma_sq)
        loss_mat = negative_MLS(mu, mu, sigma_sq, sigma_sq)
        
        label_mat = tf.equal(labels[:,None], labels[None,:])
        label_mask_pos = tf.logical_and(non_diag_mask, label_mat)
        
        loss_pos = tf.boolean_mask(loss_mat, label_mask_pos) 
    
        return tf.reduce_mean(loss_pos)


if __name__ == "__main__":

    gty = tf.convert_to_tensor([1, 2, 3, 2, 3, 3, 2])
    # muX = tf.random.normal([7, 3], mean=0, stddev=1)
    # siX = tf.random.normal([7, 3], mean=0, stddev=1)
    mu_data = np.array([[-1.7847768 , -1.0991699 ,  1.4248079 ],
                        [ 1.0405252 ,  0.35788524,  0.7338794 ],
                        [ 1.0620259 ,  2.1341069 , -1.0100055 ],
                        [-0.00963581,  0.39570177, -1.5577421 ],
                        [-1.064951  , -1.1261107 , -1.4181522 ],
                        [ 1.008275  , -0.84791195,  0.3006532 ],
                        [ 0.31099692, -0.32650718, -0.60247767]])
    
    si_data = np.array([[-0.28463233, -2.5517333 ,  1.4781238 ],
                        [-0.10505871, -0.31454122, -0.29844758],
                        [-1.3067418 ,  0.48718405,  0.6779812 ],
                        [ 2.024449  , -1.3925922 , -1.6178994 ],
                        [-0.08328865, -0.396574  ,  1.0888542 ],
                        [ 0.13096762, -0.14382902,  0.2695235 ],
                        [ 0.5405067 , -0.67946523, -0.8433032 ]])
    muX = tf.convert_to_tensor(mu_data)
    siX = tf.convert_to_tensor(si_data)
    diff = mutual_likelihood_score_loss(gty, muX, siX)
    print(diff)
    embed()
