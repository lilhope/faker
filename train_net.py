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
from lib.metric import Accuracy
from model_factory import word_factory


def parse_args():
    
    parser = argparse.ArgumentParser(description='Train Text Classification model')
    parser.add_argument('--model',help='which model used',default='CNNText',type=str)
    parser.add_argument('--gpus',help='use which gpu to train the model',default='0',type=str)
    parser.add_argument('--lr',help='the basic learning rate',default=0.0001,type=float)
    parser.add_argument('--batch_size',help='batch size',default=8,type=int)
    parser.add_argument('--momentum',help='momentum value for optimizer',default=0.9,type=float)
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--frequency',help='after n bacth output the log',default=50,type=int)
    parser.add_argument('--begin_epoch',help='beigin epoch, set to 0 means no resume',default=0,type=int)
    parser.add_argument('--end_epoch',help='end epoch',default=10,type=int)
    args = parser.parse_args()
    return args


def train_net(args):
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
    if args.begin_epoch > 0:
        logging.info('resume from epoch {}'.format(args.begin_epoch))
        net.load_params(model_path + str(args.begin_epoch).rjust(3,'0') + '.params')
    else:
        net.collect_params().initialize(init=mx.init.Xavier(),ctx=ctx)
    
    #Data Loader
    data_root = config.data_root
    dataset = os.path.join(data_root,'word_train.h5py')
    train_data = WordLoader(dataset,args.batch_size,mode='train',shuffle=False)
    val_data = WordLoader(dataset,args.batch_size,mode='val',shuffle=False)
    
    #Optimizer
    lr = args.lr
    wd = args.wd
    Trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':lr,'wd':wd})
    #Loss and evluate metirc
    metric = Accuracy(0.3)
    Loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    for epoch in range(args.begin_epoch,args.end_epoch):
        train_data.reset()
        metric.reset()
        for i,nbatch in enumerate(train_data):
            data = gluon.utils.split_and_load(nbatch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(nbatch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with ag.record():
                for x,y in zip(data,label):
                    z = net(x)
                    L = Loss(z,y)
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    L.backward()
            Trainer.step(nbatch.data[0].shape[0])
            metric.update(label,outputs)
            #output the log
            if not (i) % (args.frequency):
                name,acc = metric.get()
                logging.info('epoch[{}] batch[{}], {}={}'.format(epoch,i,name,acc))
        name,acc = metric.get()
        logging.info('epoch[%d], Training: %s=%f'%(epoch,name,acc))
        #name,val_acc = test_net(net,val_data,ctx)
        #validate
        for kbatch in val_data:
            val_data.reset()
            metric.reset()
            val_data = gluon.utils.split_and_load(kbatch.data[0], ctx_list=ctx, batch_axis=0)
            val_label = gluon.utils.split_and_load(kbatch.label[0], ctx_list=ctx, batch_axis=0)
            output = []
            for x in val_data:
                output.append(net(x))
            metric.update(val_label,output)
        name,acc = metric.get()
        logging.info('epoch[%d]m,Val: %s=%f'%(name,acc))

if __name__=="__main__":
    args = parse_args()
    train_net(args)
            
            
        
        
    
    
    
    
    
    
    
    
    
    