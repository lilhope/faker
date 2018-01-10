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
            if opt.model=='lstm':
                self.rnn = rnn.LSTM(opt.num_hidden,opt.num_layers,bidirectional=opt.bidirectional,dropout=opt.drop)
            elif opt.model == 'gru':
                self.rnn = rnn.GRU(opt.num_hidden,opt.num_layers,bidirectional=opt.bidirectional,dropout=opt.drop)
            elif opt.model == 'rnn':
                self.rnn = rnn.RNN(opt.num_hidden,opt.num_layers,dropout=opt.drop)
            else:
                raise NotImplementedError
            self.fc = nn.Dense(opt.num_hidden*2)
            self.bn = nn.BatchNorm()
            
    def forward(self,x,hidden):
        #conver NTC to TNC
        x = F.transpose(x,(1,0,2))
        output,hiddens= self.rnn(x,hidden)
        #print(output.shape)
	hidden = hiddens[-1]
	#print(hidden.shape)
	hidden = F.transpose(hidden,(1,0,2))
        output = self.fc(hidden)
        output = self.bn(output)
        output = F.relu(output)
        return output
    def begin_states(self,*args,**kwargs):
        return self.rnn.begin_state(*args,**kwargs)
"""
class RNNText(gluon.Block):
    
    def __init__(self,opt):
        super(RNNText,self).__init__()
        self.seq_len = opt.seq_len
        self.model = rnn.SequentialRNNCell()
        with self.model.name_scope():
            if opt.model =='lstm':
                self.rnn_cell = rnn.LSTMCell(opt.num_hidden)
            elif opt.model =='gru':
                self.rnn_cell = rnn.GRUCell(opt.num_hidden)
            elif opt.model == 'rnn':
                self.rnn_cell = rnn.RNNCell(opt.num_hidden)
            else:
                raise NotImplementedError
            if opt.use_res:
                self.res_cell = rnn.ResidualCell(self.rnn_cell)
            else:
                self.res_cell = None
            self.bn = nn.BatchNorm()
            if self.res_cell:
                for i in range(opt.num_layers):
                    self.model.add(self.res_cell)
                    self.model.add(rnn.DropoutCell(opt.drop))
            else:
                for i in range(opt.num_layers):
                    self.model.add(self.rnn_cell)
                    self.model.add(rnn.DropoutCell(opt.drop))
            
    def forward(self,x,states):
        #NTC to TNC
        x = F.transpose(x,(1,0,2))
        outputs = []
        for i in range(self.seq_len):
            output,states = self.model(x[i],states)
            output = self.bn(output)
            states = self.bn(states)
            outputs.append(output)
        return outputs,states
    def begin_states(self,*args,**kwargs):
        return self.model.begin_state()
"""        
if __name__=='__main__':
    opt = edict()
    opt.num_hidden= 512
    opt.seq_len = 150
    opt.model='lstm'
    opt.num_layers = 2
    opt.bidirectional= True
    opt.use_res = True
    opt.drop = 0.5
    model = RNNText(opt)
    model.collect_params().initialize()
    begin_states = model.begin_states(batch_size=20)
    x = mx.nd.random_normal(shape=(20,150,300))
    output = model(x,begin_states)
    print(output)       
        
        
    
