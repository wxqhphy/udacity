# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:55:10 2019

@author: wxq
"""

import tensorflow as tf
def get_weights(n_features,n_labels):
    return tf.Variable(tf.truncated_normal((n_features,n_labels)))
#    return tf.Variable(tf.zeros((n_features,n_labels)))

def get_biases(n_labels):
    return tf.Variable(tf.zeros(n_labels))

def linear(input,w,b):
    return tf.add(tf.matmul(input,w),b)