#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:23:04 2018

@author: lilhope
"""
#rnn的symbol接口，gluon中可以直接用gluon.rnn.LSTM()等代替
stack = mx.rnn.SequentialRNNCell()
for i in range(2):
self.stack.add(mx.rnn.GRUCell(num_hidden=512,prefix='gru_l%d'%i))

def get_rnn_feat(self,expression):
    """sequence to vector"""
    
    embed= mx.symbol.Embedding(data=expression,input_dim=12729,output_dim=300,name='embed')
    #输入是NTC，N是batch size，T是Time step，C是词向量的channel
    #按照30的长度展开，merge_output=True表示把所有time step的输出放到一个ndarray中，输出是NTC，注意gluon中默认是TNC
    outputs,states = stack.unroll(30,inputs=embed,merge_outputs=True)
    #attention
    #一个weight，也是attention机制要学习的weight
    W = mx.sym.Variable(shape=(512,),name='RNN_Att_W',init=mx.init.One())
    # NTC × C = NT，对每一个batch size的每一个time step预测一个权重
    att_dot = mx.sym.dot(outputs,W,name='RNN_Att_dot')
    #激活
    att_dot_act = mx.sym.Activation(att_dot,act_type='tanh',name='RNN_Att_dot_act')
    #预测每一个time step的权重，加起来等于1,所以用softmax
    att_weight = mx.sym.softmax(att_dot_act,name='RNN_Att_weight')
    # NT * 1，拓展一下维度，方便进行矩阵乘法
    att_weight = mx.sym.expand_dims(att_weight,axis=2,name='RNN_Att_expand')
    #NTC，每个time step的输出都乘以一个权重
    outputs = mx.sym.broadcast_mul(outputs,att_weight,name='RNN_merge')
    #NC，将每个time step的输出加起来，得到一个输出
    output = mx.sym.sum(outputs,axis=1,keepdims=False)
    #output = outputs[-1]
    #output = mx.sym.BatchNorm(data=output,eps=self.eps)
    return output
