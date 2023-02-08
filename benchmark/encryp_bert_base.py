"""
======================================================================
ENCRYP_BERT_BASE ---

Encryption of the bert-base model.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created:  7 二月 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

import math
import time

import torch
import torch.nn.functional as F
from torch import matmul as mm

import crypten
import crypten.nn as cnn
import crypten.nn as nn 
import crypten.communicator as comm

from utils import softmax_2RELU, activation_quad

from models import Bert,BertEmbeddings,BertLayer,BertAttention,BertSelfAttention,BertSelfOutput,BertIntermediate,BertOutput

class BertBaseFlatten(nn.Module):
    def __init__(self, config, timing):
        super(BertBaseFlatten,self).__init__()
        self.config=config
        self.timing=timing
        # self.config.num_labels=2
        self.device=torch.device("cuda:0")

        self.embeddings=BertEmbeddings(config,timing)

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = config.hidden_size
        self.inter_d=config.intermediate_size

        if hasattr(config,"sequence_length"):
            self.msl=config.sequence_length
        else:
            self.msl=128

        self.weight_mats=[]
        self.bias_mats=[]
        self.M_mats=[]
        ## ========================= precomputed matrix
        for num_layer in range(self.config.num_hidden_layers):
            self.weight_mats.append(torch.ones((config.hidden_size,
                                                config.hidden_size)).to(self.device))
            self.bias_mats.append(torch.ones(config.hidden_size).to(self.device))
            self.M_mats.append(torch.ones(self.num_attention_heads,
                                          self.msl,self.msl).to(self.device))
        self.last_w=torch.ones((config.hidden_size,config.hidden_size)).to(self.device)
        self.last_b=torch.ones((config.hidden_size)).to(self.device)
        
        # self.activation=activation_quad()
        self.activation=nn.ReLU()


    def multiheadMut(self,x,M):
        bs,msl,d=x.shape
        x=x.view(bs,msl,self.num_attention_heads,-1)
        x=x.transpose(1,3)

        
        for i_bs in range(bs):
            for i_shape in range(self.num_attention_heads):
                x[i_bs,:,i_shape,:]=x[i_bs,:,i_shape,:].matmul(
                                    M[i_shape])
        xo=x.transpose(1,3).reshape(bs,msl,-1)
        return xo

    def forward(self,x):
        # print(x.shape)
        xo=self.embeddings(x)

        for i in range(self.config.num_hidden_layers):
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=xo.matmul(self.weight_mats[i].T)
            xo=self.multiheadMut(xo,self.M_mats[i])
            xo=xo+self.bias_mats[i]

            c1=comm.get().get_communication_stats()
            t1=time.time()

            xo=self.activation(xo)

            c2=comm.get().get_communication_stats()
            t2=time.time()

            self.timing["LinearTime"]+=(t1-t0)
            self.timing["LinearCommTime"]+=(c1['time']-c0['time'])
            self.timing["LinearCommByte"]+=(c1['bytes']-c0['bytes'])

            self.timing["ActivTime"]+=(t2-t1)
            self.timing["ActivCommTime"]+=(c2['time']-c1['time'])
            self.timing["ActivCommByte"]+=(c2['bytes']-c1['bytes'])

        ## final transform
        t0=time.time()
        c0=comm.get().get_communication_stats()

        xo=xo.matmul(self.last_w.T)+self.last_b

        c1=comm.get().get_communication_stats()
        t1=time.time()
        self.timing["LinearTime"]+=(t1-t0)
        self.timing["LinearCommTime"]+=(c1['time']-c0['time'])
        self.timing["LinearCommByte"]+=(c1['bytes']-c0['bytes'])

        return xo

    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")
