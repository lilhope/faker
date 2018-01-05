#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:51:40 2018

@author: lilhope
model factory
"""

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn,HybridBlock
#from config import config
from model import *

class WordModel(HybridBlock):
    
    def __init__(self,feat_extractor,vocab_size=16000,embed_dim=300,num_classes=6,use_word2vec=True):
        super(WordModel,self).__init__()
        with self.name_scope():
            self.feat_extractor = feat_extractor()
            self.encoder = nn.Embedding(vocab_size,embed_dim)
            self.fc = nn.Dense(num_classes)
        if use_word2vec:
            self.encoder.
            
    def forward(self,F,x):
        encode = self.encoder(x)
        feature = self.feat_extractort(encode)
        output = self.fc(feature)
        return output
    

def word_factory(config):
    model = config.model
    opt = eval('config' + '.' + model)
    feat_extactor = eval(model)(opt)
    word_model = WordModel(feat_extactor,config.vocab_size,config.embed_dim,config.num_classes)
    return word_model
    
    
        
        
        
    