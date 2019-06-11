# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:20:25 2019

@author: wxq
"""

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn.utils
import tensorflow as tf

mnist = input_data.read_data_sets("/datasets/ud730/mnist",reshape=False)
x_train, y_train = mnist.train.images[:1000], mnist.train.labels[:1000]
x_validation, y_validation = mnist.validation.images, mnist.validation.labels
x_test, y_test = mnist.test.images, mnist.test.labels

assert(len(x_train)==len(y_train))
assert(len(x_validation)==len(y_validation))
assert(len(x_test)==len(y_test))

print("Image Shape: {}".format(x_train[0].shape))
print("Training Set: {} samples".format(len(x_train)))
print("Validation Set: {} samples".format(len(x_validation)))
print("Test Set: {} samples".format(len(x_test)))

x_train = np.pad(x_train,((0,0),(2,2),(2,2),(0,0)),"constant")
x_validation = np.pad(x_validation,((0,0),(2,2),(2,2),(0,0)),"constant")
x_test = np.pad(x_test,((0,0),(2,2),(2,2),(0,0)),"constant")

print("Updated Image Shape: {}".format(x_train[0].shape))

index = random.randint(0,len(x_train)-1)
image = x_train[index].squeeze()
plt.figure(figsize=(1,1))
plt.imshow(image,cmap="gray")
print(y_train[index])

x_train, y_train = sklearn.utils.shuffle(x_train,y_train)

epochs = 5
batch_size = 128

def LeNet(x):
    mu = 0
    sigma = 0.1
    
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5,5,1,6),mean=mu,stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_w,strides=[1,1,1,1],padding="VALID") + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    
    conv2_w = tf.Variable(tf.truncated_normal(shape=(5,5,6,16),mean=mu,stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1,conv2_w,strides=[1,1,1,1],padding="VALID")+conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    
    fc0 = tf.contrib.layers.flatten(conv2)
    
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400,120),mean=mu,stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0,fc1_w) + fc1_b
    fc1 = tf.nn.relu(fc1)
    
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120,84),mean=mu,stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)
    
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84,10),mean=mu,stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2,fc3_w) + fc3_b
    return logits

x = tf.placeholder(tf.float32,(None,32,32,1))
y = tf.placeholder(tf.int32,(None))
one_hot_y = tf.one_hot(y,10)

rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y,logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(one_hot_y,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
saver = tf.train.Saver()

def evaluate(x_data,y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0,num_examples,batch_size):
        batch_x, batch_y = x_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x:batch_x,y:batch_y})
        total_accuracy += (accuracy*len(batch_x))
    return total_accuracy/num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)
    print("Training...")
    for i in range(epochs):
        x_train, y_train = sklearn.utils.shuffle(x_train,y_train)
        for offset in range(0,num_examples,batch_size):
            end = offset + batch_size
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation,feed_dict={x:batch_x,y:batch_y})
        validation_accuracy = evaluate(x_validation,y_validation)
        print("Epoch {} ...".format(i))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    saver.save(sess,"./lenet")
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint("."))
    test_accuracy = evaluate(x_test,y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))