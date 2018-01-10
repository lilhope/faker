#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:12:47 2018

@author: lilhope
"""

import mxnet as mx
from mxnet import gluon


class FocalLoss(gluon.loss.Loss):
    
    def __init__(self,axis=-1,alpha=0.25,gamma=2,batch_axis=0,**kwargs):
        
        super(FocalLoss,self).__init__(None,batch_axis,**kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        
    def hybrid_forward(self,F,output,label):
      
        loss = -(self._alpha * ((1. - output) ** self._gamma) * F.log(output + 1e-12) * label + \
             self._alpha * ((output) ** self._gamma) * F.log(1. - output + 1e-12) * (1. - label))
        #print(loss)
        return F.mean(loss,axis = self._batch_axis,exclude=True)
    
if __name__ == '__main__':
    
    Loss = FocalLoss()
    x = mx.nd.random_normal(shape=(8,6))
    label = mx.nd.ones((8,6))
    Loss(x,label)
        
        
        
    
