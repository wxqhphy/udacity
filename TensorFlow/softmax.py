# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:12:09 2019

@author: wxq
"""

import numpy as np
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

logits = [2.0,1.0,0.1]
print(softmax(logits))

#logits = np.array([
#    [1, 2, 3, 6],
#    [2, 4, 5, 6],
#    [3, 8, 7, 6]])
#print(softmax(logits))