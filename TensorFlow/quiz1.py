# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:32:12 2019

@author: wxq
"""

import tensorflow as tf
x = tf.constant(10.)
y = tf.constant(2.)
z = tf.subtract(tf.divide(x,y),tf.constant(1.))
with tf.Session() as sess:
    output = sess.run(z)
    print(output)