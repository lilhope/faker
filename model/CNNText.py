#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 09:56:44 2018

@author: lilhope
"""

import mxnet as mx
#from config import config
from custom_layer import HybrideConcurrent
from mxnet import gluon
from mxnet.gluon import HybridBlock,nn
#import mxnet.ndarray as F

class CNNText(HybridBlock):
    def __init__(self,opt):
        super(CNNText,self).__init__()
        self.opt = opt
        with self.name_scope():
	    self.drop = nn.Dropout(opt.drop)
            #self.encoder = nn.Embedding(input_dim=opt.vocab_size,output_dim=opt.embed_dim)
            self.conv_block = HybrideConcurrent(concat_dim=1)
            for i,ngram_filter in enumerate(opt.ngram_filters):
                net = nn.HybridSequential(prefix='filter' + str(i))
                net.add(nn.Conv1D(opt.num_hidden,ngram_filter))
                #net.add(nn.BatchNorm())
                net.add(nn.Activation('relu'))
                net.add(nn.MaxPool1D(opt.seq_len - ngram_filter + 1))
                self.conv_block.add(net)
            #self.fc = nn.Dense(opt.num_classes)
            #self.bn = nn.BatchNorm()

    def hybrid_forward(self,F,x):
        #print(x)
        #embed = self.encoder(x)
        #filter_outputs = [F.relu(self.bn(conv(embed))).max(dim=2) for conv in self.Conv_layers]

        #maxpool_output = F.concat(filter_outputs,dim=1)
        x = F.transpose(x,(0,2,1))
        output = self.conv_block(x)
        output = F.flatten(output)
	#output = self.drop(output)
	#print(output)
        return output

if __name__=='__main__':
    #test
    opt = config.CNNText
    net = CNNText(opt)

    params = net.collect_params().initialize()
    x = mx.nd.random_normal(shape=(4,300,160))
    out = net(x)
    print(out)
