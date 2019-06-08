# -*- coding: utf-8 -*-
"""
Created on Sat May 25 11:08:49 2019

@author: wxq
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import quiz2
def mnist_features_labels(n_labels):
    mnist_features = []
    mnist_labels = []
    mnist = input_data.read_data_sets('/datasets/ud730/mnist',one_hot=True)
    for mnist_feature, mnist_label in zip(*mnist.train.next_batch(10000)):
        if mnist_label[:n_labels].any():
            mnist_features.append(mnist_feature)
            mnist_labels.append(mnist_label[:n_labels])
    return mnist_features, mnist_labels

n_features = 784
n_labels = 3

features = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)
w = quiz2.get_weights(n_features,n_labels)
b = quiz2.get_biases(n_labels)
logits = quiz2.linear(features,w,b)
prediction = tf.nn.softmax(logits)
cross_entropy = -tf.reduce_sum(labels*tf.log(prediction),reduction_indices=1)
loss = tf.reduce_mean(cross_entropy)
learning_rate = 0.08
optimezer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train_features, train_labels = mnist_features_labels(n_labels)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
#    prediction = tf.nn.softmax(logits)
#    cross_entropy = -tf.reduce_sum(labels*tf.log(prediction),reduction_indices=1)
#    loss = tf.reduce_mean(cross_entropy)
#    learning_rate = 0.08
#    optimezer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    res_opti, res_loss = session.run([optimezer,loss],feed_dict={features:train_features,labels:train_labels})

print('Loss: {}'.format(res_loss))