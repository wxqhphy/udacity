# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:28:40 2019

@author: wxq
"""

import tensorflow as tf

def run():
    output = None
    logit_data = [2.0,1.0,0.1]
    logits = tf.placeholder(tf.float32)
    softmax = tf.nn.softmax(logits)
    
    with tf.Session() as sess:
        output = sess.run(softmax,feed_dict={logits:logit_data})
    
    return output

print(run())