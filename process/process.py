#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:21:09 2018
data preprocess
@author: lilhope
"""
import os
import pandas as pd
import numpy as np
import h5py
from text import Tokenizer
from sequence import pad_sequences

def process_word(sents_train,sents_test,embed_file):
    #tokenizer
    tokenizer = Tokenizer(num_words=16000)
    tokenizer.fit_on_texts(sents_train)
    tokened_train = tokenizer.texts_to_sequences(sents_train)
    tokened_test = tokenizer.texts_to_sequences(sents_test)
    #get the glovec embed
    word_embeding = dict()
    with open(embed_file) as f:
        line = f.readline()
        data = line.strip().split()
        word = data[0]
        vec = np.array(data[1:],dtype=np.float32)
        word_embeding[word] = vec
    #get the embedding metric
    #TODO: get the means and stds 
    embed_metric = np.random.normal(0.020940498, 0.6441043,(16000,300))
    
    for word,idx in tokenizer.word_index.iteritems():
        embed = word_embeding.get(word)
        if embed is not None:
            embed_metric[idx] = embed
    return tokened_train,tokened_test,embed_metric

def process_char():
    pass


if __name__ == "__main__":
    train_data = os.path.join('../data','train.csv')
    test_data = os.path.join('../data','test.csv')
    glo_vec = os.path.join('../data','glove.840B.300d.txt')
    df_train = pd.read_csv(train_data)
    df_test = pd.read_csv(test_data)
    sents_train = df_train['comment_text'].fillna('_na_').values
    sents_test = df_test['comment_text'].fillna('_na_').values
    """
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(sents_train)
    word_count = tokenizer.word_counts
    word_docs = tokenizer.word_docs
    tokened_train = tokenizer.texts_to_sequences(sents_train)
    tokened_test = tokenizer.texts_to_sequences(sents_test)
    """
    tokened_train,token_test,embed_metric = process_word(sents_train,sents_test,glo_vec)
    #pad to same length
    tokened_train = pad_sequences(tokened_train,maxlen=150,truncating='post')
    token_test = pad_sequences(token_test,maxlen=150,truncating='post')
    classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    labels = df_train[classes].values
    #writ to the h5py file
    File = h5py.File('../data/word_train.h5py','w')
    File.create_dataset('data',data=np.array(tokened_train,dtype=np.float32))
    File.create_dataset('label',data=np.array(labels,dtype=np.float32))
    File.close()
    File = h5py.File('../data/word_test.h5py','w')
    File.create_dataset('data',data=np.array(token_test,dtype=np.float32))
    File.close()
    np.save('../data/word_embed.npy',embed_metric)
    
    
