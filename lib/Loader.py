#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:00:02 2018
Data Loader for mxnet
@author: lilhope
"""
import mxnet as mx
import h5py
import numpy as np


class WordLoader(mx.io.DataIter):
    
    def __init__(self,dataset,batch_size,mode='train',shuffle=False):
        
        self.batch_size = batch_size
        dataset = h5py.File(dataset)
        self.mode = mode
        self.data_name = ['data']
        #self.data = dataset['data']
        #self.label = dataset['label'].value
        if self.mode=='train':
            self.label_name = ['label']
            self.data = dataset['data'].value[:86000]
            self.label = dataset['label'].value[:86000]
        elif self.mode=='val':
            self.label_name = ['label']
            self.data = dataset['data'].value[86000:]
            self.label = dataset['label'].value[86000:]
        else:
            self.label_name=None
            self.data = dataset['data'].value
        dataset.close()
        self.shuffle=shuffle
        self.size = self.data.shape[0]
        print(self.size)
        self.index = np.arange(self.size)
        self.cur = 0
    @property
    def provide_data(self):
        return [(k,v.shape) for k,v in zip(self.data_name,self._data)]    
    @property
    def provide_label(self):
        if not self.mode=='test':
            return [(k,v.shape) for k,v in zip(self.label_name,self._label)]
        else:
            return None
    
    def get_pad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0
        
    def get_index(self):
        return self.cur / self.batch_size
    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)
    def iter_next(self):
        return self.cur + self.batch_size <= self.size
    
    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self._data,label=self._label,
                                   pad = self.getpad(),index=self.getindex(),
                                   provide_data=self.provide_data,provide_label=self.provide_label)
        else:
            raise StopIteration
    def get_batch(self):
        #get the data
        cur_from = self.cur
        cur_to = self.cur + self.batch_size
        index = self.index[cur_from:cur_to]
        self._data = [mx.nd.array(self.data[index])]
        if not self.mode=='test':
            self._label = [mx.nd.array(self.label[index])]
        else:
            self._label = None
if __name__=="__main__":
    data = './data/word_train.h5py'
    loader = WordLoader(data,4,is_train=True)
    #x = loader.next()
    for x in loader:
        print(x.provide_data)
