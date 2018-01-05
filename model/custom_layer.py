#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:13:50 2018

@author: lilhope
"""
import mxnet as mx
from mxnet.gluon import HybridBlock

class HybrideConcurrent(HybridBlock):
    
    def __init__(self,concat_dim,prefix=None,params=None):
        super(HybrideConcurrent,self).__init__(prefix=prefix,params=params)
        self.concat_dim = concat_dim
        
    def add(self,block):
        self.register_child(block)
    
    def hybrid_forward(self,F,x):
        out = []
        for block in self._children:
            out.append(block(x))
        #print(out)
        out = F.concat(*out,dim=self.concat_dim)
        return out