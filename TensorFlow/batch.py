# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:24:30 2019

@author: wxq
"""

#import pprint

def batches(batch_size,features,labels):
    assert len(features)==len(labels)
    output_batches = []
    sample_size = len(features)
    #print(sample_size)
    for start_i in range(0,sample_size,batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i],labels[start_i:end_i]]
        output_batches.append(batch)
    return output_batches

#example_features = [
#    ['F11','F12','F13','F14'],
#    ['F21','F22','F23','F24'],
#    ['F31','F32','F33','F34'],
#    ['F41','F42','F43','F44']]
#
#example_labels = [
#    ['L11','L12'],
#    ['L21','L22'],
#    ['L31','L32'],
#    ['L41','L42']]
#
#pprint.pprint(batches(3, example_features, example_labels))