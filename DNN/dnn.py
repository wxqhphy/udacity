# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 21:05:14 2019

@author: wxq
"""

import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/datasets/ud730/mnist',one_hot=True)
learning_rate = 0.001
training_epochs = 20
batch_size = 128
total_batch = math.ceil(mnist.train.num_examples/batch_size)
n_input = 784
n_classes = 10
n_hidden_layer = 256

weights = {
    'hidden_layer':tf.Variable(tf.random_normal([n_input,n_hidden_layer])),
    'out':tf.Variable(tf.random_normal([n_hidden_layer,n_classes]))
    }
biases = {
    'hidden_layer':tf.Variable(tf.random_normal([n_hidden_layer])),
    'out':tf.Variable(tf.random_normal([n_classes]))
    }

x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_classes])

layer_1 = tf.add(tf.matmul(x,weights['hidden_layer']),biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)
logits = tf.add(tf.matmul(layer_1,weights['out']),biases['out'])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            res_opti,res_cost = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
        print("epoch %d, batch %d, cost %f" % (epoch,i,res_cost))