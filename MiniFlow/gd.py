# -*- coding: utf-8 -*-
"""
Created on Sat May  4 09:16:16 2019

@author: wxq
"""

def gradient_descent_update(x,gradx,learning_rate):
    x = x - learning_rate * gradx
    return x