# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:17:58 2019

@author: wxq
"""

import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
learning_rate = 0.001
n_input = 784
n_classes = 10
batch_size = 128
n_epochs = 100
mnist = input_data.read_data_sets('/datasets/ud730/mnist',one_hot=True)
save_file = "./train_model.ckpt"

features = tf.placeholder(tf.float32,[None,n_input])
labels = tf.placeholder(tf.float32,[None,n_classes])
weights = tf.Variable(tf.random_normal([n_input,n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))
logits = tf.add(tf.matmul(features,weights),bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        total_batch = math.ceil(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_features, batch_labels = mnist.train.next_batch(batch_size)
            sess.run(optimizer,feed_dict={features:batch_features,labels:batch_labels})
        if epoch % 10 == 0:
            valid_accuracy = sess.run(accuracy,feed_dict={features:mnist.validation.images,labels:mnist.validation.labels})
            print("Epoch {:<3} - Vlidation Accuracy: {}".format(epoch,valid_accuracy))
    test_accuracy = sess.run(accuracy,feed_dict={features:mnist.test.images,labels:mnist.test.labels})
    saver.save(sess,save_file)

print("Test Accuracy: {}".format(test_accuracy))
print("Trained Model Saved.")

with tf.Session() as sess:
    saver.restore(sess,save_file)
    test_accuracy = sess.run(accuracy,feed_dict={features:mnist.test.images,labels:mnist.test.labels})

print("Test Accuracy: {}".format(test_accuracy))