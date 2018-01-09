#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:51:40 2018

@author: lilhope
model factory
"""
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn,HybridBlock,Block
import mxnet.ndarray as f
from config import config as default
from model import *

class WordModel(HybridBlock):
    
    def __init__(self,feat_extractor,ctx,vocab_size=16000,embed_dim=300,num_classes=6,use_word2vec=True):
        super(WordModel,self).__init__()
        with self.name_scope():
            self.feat_extractor = feat_extractor
            self.encoder = nn.Embedding(vocab_size,embed_dim)
            self.fc = nn.Dense(num_classes)
        self.encoder.initialize(ctx=ctx)
        if use_word2vec:
            self.encoder.weight.set_data(mx.nd.array(np.load(default.w2v_file)))
            
            
    def hybrid_forward(self,F,x):
        encode = self.encoder(x)
        feature = self.feat_extractor(encode)
        output = self.fc(feature)
        output = F.sigmoid(output)
        return output

class wordmodel(Block):
    
    def __init__(self,feat_extractor,ctx,vocab_size=16000,embed_dim=300,num_classes=6,use_word2vec=True):
        super(wordmodel,self).__init__()
        with self.name_scope():
            self.feat_extractor = feat_extractor
            self.encoder = nn.Embedding(vocab_size,embed_dim)
            self.fc = nn.Dense(num_classes)
        self.encoder.initialize(ctx=ctx)
        if use_word2vec:
            self.encoder.weight.set_data(mx.nd.array(np.load(default.w2v_file)))
            
            
    def forward(self,x,begin_states):
        encode = self.encoder(x)
        feature = self.feat_extractor(encode,begin_states)
        output = self.fc(feature)
        output = f.sigmoid(output)
        return output
    

def word_factory(config):
    print(config)
    model = config.model
    opt = eval('config' + '.' + model)
    print(model)
    feat_extactor = eval(model)(opt)
    print(type(feat_extactor))
    if isinstance(feat_extactor,HybridBlock):    
        word_model = WordModel(feat_extactor,config.ctx,config.vocab_size,config.embed_dim,config.num_classes)
    else:
        word_model = wordmodel(feat_extactor,config.ctx,config.vocab_size,config.embed_dim,config.num_classes)
    return word_model

if __name__=="__main__":
    net = word_factory(default)
    net.collect_params().initialize()
    x = mx.nd.ones((4,160))
    y = net(x)
    print(y)
    
    
        
        
        
    