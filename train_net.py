#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:49:57 2018

@author: lilhope
Trainer
"""
import mxnet as mx
from mxnet import gluon
import argparse
import logging
from lib import Loader
from model_factory import word_factory


def parse_args():
    
    parser = argparse.ArgumentParser(description='Train Text Classification model')
    parser.add_argument('--model',help='which model used',default='CNNText',type=str)
    parser.add_argument('--gpus',help='use which gpu to train the model',default='1,4',type=str)
    parser.add_argument('--lr',help='the basic learning rate',deault=0.001,type=float)
    parser.add_argument('--momentum',help='momentum value for optimizer',default=0.9,type=float)
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--frequency',help='after n bacth output the log',default=50,type=int)
    args = parser.parse_args()
    return args


def train_net(args):
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus else [mx.cpu()]
    config.model = args.model
    #get the model
    net = word_factory(config)
    
    
    
    
    
    
    
    