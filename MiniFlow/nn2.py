# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:41:19 2019

@author: wxq
"""

import numpy as np
import sklearn.datasets
import sklearn.utils
import miniflow

data = sklearn.datasets.load_boston()
x_ = data['data']
y_ = data['target']

x_ = (x_-np.mean(x_,axis=0))/np.std(x_,axis=0)
n_features = x_.shape[1]
n_hidden = 10
w1_ = np.random.randn(n_features,n_hidden)
b1_ = np.zeros(n_hidden)
w2_ = np.random.randn(n_hidden,1)
b2_ = np.zeros(1)

x, y = miniflow.Input(), miniflow.Input()
w1, b1 = miniflow.Input(), miniflow.Input()
w2, b2 = miniflow.Input(), miniflow.Input()
l1 = miniflow.Linear(x,w1,b1)
s1 = miniflow.Sigmoid(l1)
l2 = miniflow.Linear(s1,w2,b2)
cost = miniflow.MSE(y,l2)
feed_dict = {x:x_, y:y_, w1:w1_, b1:b1_, w2:w2_, b2:b2_}
epochs = 10

m = x_.shape[0]
batch_size = 11
steps_per_epoch = m // batch_size
graph = miniflow.topological_sort(feed_dict)
trainables = [w1,b1,w2,b2]
print("Total number of examples = {}".format(m))

for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        x_batch, y_batch = sklearn.utils.resample(x_,y_,n_samples=batch_size)
        x.value = x_batch
        y.value = y_batch
        miniflow.forward_and_backward(graph)
        miniflow.sgd_update(trainables)
        loss += graph[-1].value
    print("Epoch: {}, Loss: {:.3f}".format(i+1,loss/steps_per_epoch))