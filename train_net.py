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
from lib.loss import FocalLoss
from model_factory import word_factory
from sklearn.model_selection import train_test_split
import h5py
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser(description='Train Text Classification model')
    parser.add_argument('--model',help='which model used',default='CNNText',type=str)
    parser.add_argument('--gpus',help='use which gpu to train the model',default='0',type=str)
    parser.add_argument('--lr',help='the basic learning rate',default=0.001,type=float)
    parser.add_argument('--batch_size',help='batch size',default=8,type=int)
    parser.add_argument('--momentum',help='momentum value for optimizer',default=0.9,type=float)
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr_step',help='learning rate step',default='3,5',type=str)
    parser.add_argument('--lr_factor',help='learning rate deacy factor',default=0.1,type=float)
    parser.add_argument('--frequency',help='after n bacth output the log',default=50,type=int)
    parser.add_argument('--begin_epoch',help='beigin epoch, set to 0 means no resume',default=0,type=int)
    parser.add_argument('--end_epoch',help='end epoch',default=10,type=int)
    args = parser.parse_args()
    return args

def parse_log(names,values):
    info = ''
    for name,value in zip(names,values):
        info += "{} = {} \t".format(name,value)
    return info

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
        net.load_params(model_path + '/' + str(args.begin_epoch).rjust(3,'0') + '.params',ctx=ctx)
    else:
        net.collect_params().initialize(init=mx.init.Xavier(),ctx=ctx)
    if config.begin_states:
        begin_states = net.feat_extractor.begin_states(batch_size=args.batch_size,func = mx.nd.zeros,ctx=ctx[0])
    else:
        begin_states = None

    #Data Loader
    data_root = config.data_root
    dataset = h5py.File(os.path.join(data_root,'word_train.h5py'))
    data = dataset['data'].value
    label = dataset['label'].value
    any_cat_pos = np.sum(label,1)
    data_train,data_val,label_train,label_val = train_test_split(data,label,
								test_size=0.2,
								stratify = any_cat_pos,
								random_state=2018)
    print(data_train.shape)
    print(label_train.shape)

    train_data = WordLoader(data_train,args.batch_size,mode='train',label=label_train,shuffle=True)
    val_data = WordLoader(data_val,args.batch_size,mode='val',label=label_val,shuffle=False)

    #Optimizer
    lr = args.lr
    lr_step =  [int(i) for i in args.lr_step.split(',')]
    wd = args.wd
    Trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':lr,'wd':wd})
    #Loss and evluate metirc
    metric = mx.metric.CompositeEvalMetric()
    #loss_metric = LogLoss()
    metric.add(Accuracy(0.5))
    metric.add(LogLoss())
    #Loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    Loss = FocalLoss()
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
                    if begin_states:
                        z = net(x,begin_states)
                        
                    else:
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
                names,values = metric.get()
                info = parse_log(names,values)
                logging.info('epoch[{}] batch[{}]'.format(epoch,i) + info )
        names,values = metric.get()
        info = parse_log(names,values)
        logging.info('epoch[{}], Training '.format(epoch) + info)
        #name,val_acc = test_net(net,val_data,ctx)
        #validate
        val_data.reset()
        for kbatch in val_data:
            metric.reset()
            valdata = gluon.utils.split_and_load(kbatch.data[0], ctx_list=ctx, batch_axis=0)
            vallabel = gluon.utils.split_and_load(kbatch.label[0], ctx_list=ctx, batch_axis=0)
            output = []
            for x in valdata:
                if begin_states:
                    z = net(x,begin_states)
                else:
                    z = net(x)
                output.append(z)
            metric.update(vallabel,output)
        names,values = metric.get()
        info = parse_log(names,values)
        logging.info('epoch[{}],Val'.format(epoch) + info)
        #reset the learning rate
        if epoch in lr_step:
            lr = lr * args.lr_factor
            Trainer.set_learning_rate(lr)
            logging.info('changing learning rate to {}'.format(lr))
        #save the model
        net.collect_params().save(model_path + '/' + str(epoch).rjust(3,'0') + '.params')

if __name__=="__main__":
    args = parse_args()
    train_net(args)
