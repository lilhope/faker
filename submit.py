#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:49:57 2018

@author: lilhope
Trainer
"""
import os
import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag
import argparse
import logging
from config import config
from lib.Loader import WordLoader
from lib.metric import Accuracy,LogLoss
from model_factory import word_factory


def parse_args():
    
    parser = argparse.ArgumentParser(description='Train Text Classification model')
    parser.add_argument('--model',help='which model used',default='CNNText',type=str)
    parser.add_argument('--gpus',help='use which gpu to train the model',default='0',type=str)
    parser.add_argument('--epoch',help='end epoch',default=9,type=int)
    args = parser.parse_args()
    return args

def parse_log(names,values):
    info = ''
    for name,value in zip(names,values):
        info += "{} = {} \t".format(name,value)
    return info

def sumbit(args):
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus else [mx.cpu()]
    config.model = args.model
    config.ctx = ctx
    model_path = os.path.join(config.result_path,config.model)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    #get the model
    net = word_factory(config)
    logging.info('submit the result from epoch {}'.format(args.epoch))
    net.collect_params().load(model_path + '/' + str(args.epoch).rjust(3,'0') + '.params',ctx=ctx)
    
    
    #Data Loader
    data_root = config.data_root
    dataset = os.path.join(data_root,'word_test.h5py')
    test_data = WordLoader(dataset,1,mode='test',shuffle=False)
    test_data.reset()
    output = []
    for kbatch in test_data:
        valdata = gluon.utils.split_and_load(kbatch.data[0], ctx_list=ctx, batch_axis=0)
        for x in valdata:
            output.append(net(x))
    
    print(output)

if __name__=="__main__":
    args = parse_args()
    sumbit(args)
            
            
        
        
    
    
    
    
    
    
    
    
    
    
