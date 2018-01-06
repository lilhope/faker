#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:17:19 2018

@author: lilhope
evaluation metric during training
"""

import mxnet as mx
import numpy as np

class Accuracy(mx.metric.EvalMetric):
    
    def __init__(self,thresh):
        super(Accuracy,self).__init__('Accuarcy')
        self.thresh = thresh
    
    def update(self,labels,preds):
        
        for label,pred in zip(labels,preds):
            pred_label = pred.asnumpy()
            pred_label[pred_label >= self.thresh] = 1.
            pred_label[pred_label < self.thresh] = 0.
            #print(pred_label)
            label = label.asnumpy()
            
            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)
