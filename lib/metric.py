#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:17:19 2018

@author: lilhope
evaluation metric during training
"""

import mxnet as mx
import numpy as np
from sklearn.metrics import log_loss

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

class LogLoss(mx.metric.EvalMetric):
    
    def __init__(self,eps=1e-5):
        super(LogLoss,self).__init__('LogLoss')
        self.eps = eps
        
    def update(self,labels,preds):
        
        for label,pred in zip(labels,preds):
            label = label.asnumpy()
            pred = pred.asnumpy()
            loss = np.zeros_like(pred)
            """
            pos_indexs = np.where(label==1)
            neg_indexs = np.where(label==1)
            loss[pos_indexs] = -np.log(pred[pos_indexs])
            loss[neg_indexs] = -np.log(1 - pred[neg_indexs])
            """
            loss = -(label * np.log(pred + self.eps) + (1 - label) * np.log(1 - pred + self.eps))
            self.sum_metric += np.sum(loss)
            self.num_inst += len(pred.flat)
            
