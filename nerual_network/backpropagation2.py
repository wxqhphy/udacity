# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:50:16 2019

@author: wxq
"""

import numpy as np
from data_prep import features, targets, features_test, targets_test
np.random.seed(42)
def sigmoid(x):
    return 1/(1+np.exp(-x))

n_hidden = 2
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None

weights_input_hidden = np.random.normal(scale=1/n_features**.5,size=(n_features,n_hidden))
weights_hidden_output = np.random.normal(scale=1/n_features**.5,size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x,y in zip(features.values,targets):
        hidden_output = sigmoid(np.dot(x,weights_input_hidden))
        output = sigmoid(np.dot(hidden_output,weights_hidden_output))
        
        error = y - output
        output_error = error * output * (1 - output)
        hidden_error = np.dot(output_error,weights_hidden_output) * hidden_output * (1 - hidden_output)
        
        del_w_hidden_output += output_error * hidden_output
        del_w_input_hidden += hidden_error * x[:,None]

    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records
    
    if e % (epochs/10) == 0:
        hidden_output = sigmoid(np.dot(x,weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,weights_hidden_output))
        loss = np.mean((out-targets)**2)
        if last_loss and last_loss < loss:
            print("Train loss: ",loss," WARNING - Loss Increasing")
        else:
            print("Train loss: ",loss)
        last_loss = loss

hidden = sigmoid(np.dot(features_test,weights_input_hidden))
out = sigmoid(np.dot(hidden,weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions==targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))