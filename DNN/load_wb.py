# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 15:55:23 2019

@author: wxq
"""

import tensorflow as tf

tf.reset_default_graph()

save_file = "./model.ckpt"
weights = tf.Variable(tf.truncated_normal([2,3]),name="weights_0")
bias = tf.Variable(tf.truncated_normal([3]),name="bias_0")
saver = tf.train.Saver()

print("Save Weights: {}".format(weights.name))
print("Save Bias: {}".format(bias.name))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess,save_file)

tf.reset_default_graph()

bias = tf.Variable(tf.truncated_normal([3]),name="bias_0")
weights = tf.Variable(tf.truncated_normal([2,3]),name="weights_0")
saver = tf.train.Saver()

print("Load Weights: {}".format(weights.name))
print("Load Bias: {}".format(bias.name))

with tf.Session() as sess:
    saver.restore(sess,save_file)