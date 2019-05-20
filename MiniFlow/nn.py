# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:06:25 2019

@author: wxq
"""

import numpy as np
import miniflow
#x,y = miniflow.Input(), miniflow.Input()
#f = miniflow.Add(x,y)
#feed_dict = {x:10, y:5}
#sorted_nodes = miniflow.topological_sort(feed_dict)
#output = miniflow.forward_pass(f,sorted_nodes)
#print("{} + {} = {} (according to miniflow)".format(feed_dict[x],feed_dict[y],output))

#X, W, b = miniflow.Input(), miniflow.Input(), miniflow.Input()
#f = miniflow.Linear(X,W,b)
#g = miniflow.Sigmoid(f)
#X_ = np.array([[-1.,-2.],[-1.,-2.]])
#W_ = np.array([[2.,-3.],[2.,-3.]])
#b_ = np.array([-3.,-5.])
#feed_dict = {X:X_, W:W_, b:b_}
#graph = miniflow.topological_sort(feed_dict)
#output = miniflow.forward_pass(g,graph)
#print(output)

#y,a = miniflow.Input(), miniflow.Input()
#cost = miniflow.MSE(y,a)
#y_ = np.array([1,2,3])
#a_ = np.array([4.5,5,10])
#feed_dict = {y:y_, a:a_}
#graph = miniflow.topological_sort(feed_dict)
#miniflow.forward_pass(graph)
#print(cost.value)

X, W, b, y = miniflow.Input(), miniflow.Input(), miniflow.Input(), miniflow.Input()
f = miniflow.Linear(X,W,b)
a = miniflow.Sigmoid(f)
cost = miniflow.MSE(y,a)

X_ = np.array([[-1.,-2.],[-1.,-2.]])
W_ = np.array([[2.],[3.]])
b_ = np.array([-3.])
y_ = np.array([1.,2.])

feed_dict = {X:X_, y:y_, W:W_, b:b_}

graph = miniflow.topological_sort(feed_dict)
miniflow.forward_and_backward(graph)
gradients = [t.gradients[t] for t in [X,y,W,b]]
print(gradients)