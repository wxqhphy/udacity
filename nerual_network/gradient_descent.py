# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:42:36 2019

@author: wxq
"""

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

learnrate = 0.5
x = np.array([1,2])
y = 0.5
w = np.array([0.5,-0.5])

nn_output = sigmoid(np.dot(x,w))
error = y - nn_output
del_w = learnrate * error * sigmoid_prime(np.dot(x,w)) * x

print('Neural NetWork output:',nn_output)
print('Amount of Error:',error)
print('Change in Weights:',del_w)