# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 06:45:52 2019

@author: wxq
"""

import numpy as np
from data_prep import features, targets, features_test, targets_test

def sigmoid(x):
    return 1/(1+np.exp(-x))

np.random.seed(42)

n_records, n_features = features.shape
last_loss = None
weights = np.random.normal(scale=1/n_features**.5,size=n_features)
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x,y in zip(features.values,targets):
        output = sigmoid(np.dot(x,weights))
        error = y - output
        del_w += error * output * (1-output) * x
    weights += learnrate * del_w / n_records
    
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features,weights))
        loss = np.mean((out-targets)**2)
        if last_loss and last_loss < loss:
            print("Train loss: ",loss," WARNING - Loss Increasing")
        else:
            print("Train loss: ",loss)
        last_loss = loss

test_out = sigmoid(np.dot(features_test,weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions==targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))