# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:22:30 2019

@author: wxq
"""

import pickle
import numpy as np
#import tensorflow as tf
#from tensorflow.python.ops import control_flow_ops
#control_flow_ops = tf

with open("small_train_traffic.p",mode="rb") as f:
    data = pickle.load(f)
x_train, y_train = data["features"], data["labels"]

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#model.add(Flatten(input_shape=(32,32,3)))
model.add(Conv2D(32,(3,3),input_shape=(32,32,3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(5))
model.add(Activation("softmax"))

x_normalized = np.array(x_train/255.0-0.5)

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile("adam","categorical_crossentropy",["accuracy"])
history = model.fit(x_normalized,y_one_hot,epochs=10,validation_split=0.2)

with open("small_test_traffic.p","rb") as f:
    data_test = pickle.load(f)

x_test = data_test["features"]
y_test = data_test["labels"]

x_normalized_test = np.array(x_test/255.0-0.5)
y_one_hot_test = label_binarizer.fit_transform(y_test)

print("Testing")

metrics = model.evaluate(x_normalized_test,y_one_hot_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print("{}: {}".format(metric_name,metric_value))