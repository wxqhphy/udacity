# -*- coding: utf-8 -*-
"""
Created on Sat May  4 09:07:19 2019

@author: wxq
"""

import random
import gd

def f(x):
    return x**2 + 5

def df(x):
    return 2*x

x = random.randint(0,10000)
learning_rate = 0.1
epochs = 100

for i in range(epochs+1):
    cost = f(x)
    gradx = df(x)
    print("EPOCH {}: Cost = {:.3f}, x = {:.3f}".format(i,cost,gradx))
    x = gd.gradient_descent_update(x,gradx,learning_rate)