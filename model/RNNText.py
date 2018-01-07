#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 10:40:59 2018

@author: lilhope
"""
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn,rnn,Block
import mxnet.ndarray as F
from easydict import EasyDict as edict

class RNNText(gluon.Block):
    
    def __init__(self,opt):
        super(RNNText,self).__init__()
        with self.name_scope():
            if opt.mode=='lstm':
                self.rnn = rnn.LSTM(opt.num_hidden,opt.num_layers,bidirectional=opt.bidirectional)
            elif opt.mode == 'gru':
                self.rnn = rnn.GRU(opt.num_hidden,opt.num_layers,bidirectional=opt.bidirectional)
            elif opt.mode == 'rnn':
                self.rnn = rnn.RNN(opt.num_hidden,opt.num_layers)
            else:
                raise NotImplementedError
            self.fc = nn.Sequential()
            self.fc.add(nn.Dense(opt.num_hidden*2))
            self.fc.add(nn.BatchNorm())
            self.fc.add(nn.Activation('relu'))
    def forward(self,x,hidden):
        #conver NTC to TNC
        x = F.transpose(x,(1,0,2))
        output,hidden = self.rnn(x,hidden)
        #print(hidden)
        output = self.fc(hidden[-1])
        return output
    def begin_states(self,*args,**kwargs):
        return self.rnn.begin_state(*args,**kwargs)
if __name__=='__main__':
    opt = edict()
    opt.num_hidden= 512
    opt.mode='lstm'
    opt.num_layers = 2
    opt.bidirectional= True
    model = RNNText(opt)
    print(type(model))
    model.collect_params().initialize()
    begin_states = model.begin_states(func=mx.nd.zeros,batch_size=4)
    hidden = [i.detach() for i in begin_states]
    #print(hidden)
    input_ = mx.nd.random_normal(shape=(4,10,300))
    output = model(input_,hidden)
    print(output)
        
        
        
    
