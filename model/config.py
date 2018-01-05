#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:04:06 2018

@author: lilhope
"""
import numpy as np
from easydict import EasyDict as edict

config = edict()
config.data_root = './data'
config.result_path = './result'
config.vocab_size = 16000
config.embeb_dim = 300

config.CNNText = edict()
config.CNNText.num_hidden= 256
config.CNNText.ngram_filters = [2,3,4,5]
config.CNNText.vocab_size = 16000
config.CNNText.embed_dim = 300
config.CNNText.seq_len = 160
