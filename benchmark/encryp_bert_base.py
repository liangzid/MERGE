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

from utils import softmax_2RELU, activation_quad, softmax2RELU_2

from models import Bert,BertEmbeddings,BertLayer,BertAttention,BertSelfAttention,BertSelfOutput,BertIntermediate,BertOutput

class BertBaseFlatten(nn.Module):
    def __init__(self, config, timing):
        super(BertBaseFlatten,self).__init__()
        self.config=config
        self.timing=timing
        # self.config.num_labels=2
        self.device=torch.device(config.device)
        device=self.device

        self.embeddings=BertEmbeddings(config,timing)
        self.embeddings.cuda()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = config.hidden_size
        self.inter_d=config.intermediate_size

        if hasattr(config,"sequence_length"):
            self.msl=config.sequence_length
        else:
            self.msl=128

        self.weight1_mats=[]
        self.weight2_mats=[]
        self.bias_mats=[]
        self.M_mats=[]
        self.I=config.intermediate_size
        self.d=config.hidden_size
        ## ========================= precomputed matrix
        self.bgin_layer=nn.Linear(config.hidden_size,self.I)
        self.bgin_w=torch.ones((self.I,config.hidden_size)).to(device).T
        self.bgin_b=torch.ones(self.I).to(device)
        self.bgin_M=torch.ones(self.num_attention_heads,
                               self.msl,self.msl).to(device)
        self.bgin_w=crypten.cryptensor(self.bgin_w,src=0)
        self.bgin_b=crypten.cryptensor(self.bgin_b,src=0)
        self.bgin_M=crypten.cryptensor(self.bgin_M,src=0)
        
        self.d_ls=cnn.ModuleList([
            cnn.Linear(self.I,self.I) for _ in\
            range(config.num_hidden_layers-1)
            ])
        for num_layer in range(self.config.num_hidden_layers-1):
            
            self.weight1_mats.append(torch.ones((self.d,
                                                self.I)).to(self.device).T)
            self.weight2_mats.append(torch.ones((self.I,
                                                self.d)).to(self.device).T)
            self.bias_mats.append(torch.ones(self.I).to(self.device))
            self.M_mats.append(torch.ones(self.num_attention_heads,
                            self.msl,self.msl).to(self.device))

        for i in range(self.config.num_hidden_layers-1):
            self.weight1_mats[i]=crypten.cryptensor(self.weight1_mats[i],
                                                src=0)
            self.weight2_mats[i]=crypten.cryptensor(self.weight2_mats[i],
                                                src=0)
            self.bias_mats[i]=crypten.cryptensor(self.bias_mats[i],
                                                src=0)
            self.M_mats[i]=crypten.cryptensor(self.M_mats[i],
                                                src=0)

        self.f_layer=nn.Linear(self.I,config.hidden_size)
        self.last_w=torch.ones((config.hidden_size,
                                self.I)).to(self.device).T
        self.last_b=torch.ones((config.hidden_size)).to(self.device)

        self.last_w=crypten.cryptensor(self.last_w,src=0)
        self.last_b=crypten.cryptensor(self.last_b,src=0)
        
        self.activation=activation_quad()
        # self.activation=nn.ReLU()

        # self.d_ls=nn.ModuleList([
        #     nn.Linear(config.hidden_size,config.hidden_size) for _ \
        #     in range(self.config.num_hidden_layers)
        #     ])
        # self.M_ls=nn.ModuleList([
        #     nn.Linear()
        #     ])


    def multiheadMut(self,x,M):
        bs,msl,d=x.shape 
        x=x.view(bs,msl,self.num_attention_heads,-1)
        x=x.permute(0,2,3,1) # 1,d/num_head,num_head,msl

        # expect M: bs,num_head,msl,1
        d=M.shape[-1]
        M=M.unsqueeze(0)
        if bs>1:
            M=M.repeat(bs)
        # M=M.repeat(1,1,1,msl)

        xo=x.matmul(M)

        # bs,msl,d/num_head,num_head
        if d==1:
            xo=xo.permute(0,2,3,1).reshape(bs,1,-1)
        else:
            xo=xo.permute(0,2,3,1).reshape(bs,msl,-1)
        return xo

    def forward(self,x):
        # print(x.shape)
        xo=self.embeddings(x)

        t0=time.time()
        c0=comm.get().get_communication_stats()

        xo=xo.matmul(self.bgin_w)
        xo=self.multiheadMut(xo,self.bgin_M)
        xo=xo+self.bgin_b

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
        

        for i in range(self.config.num_hidden_layers-1):
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=xo.matmul(self.weight1_mats[i])
            xo=xo.matmul(self.weight2_mats[i])
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

        xo=xo.matmul(self.last_w)+self.last_b

        c1=comm.get().get_communication_stats()
        t1=time.time()
        self.timing["LinearTime"]+=(t1-t0)
        self.timing["LinearCommTime"]+=(c1['time']-c0['time'])
        self.timing["LinearCommByte"]+=(c1['bytes']-c0['bytes'])

        return xo

    def forwardLayer(self,x):
        # print(x.shape)
        xo=self.embeddings(x)

        t0=time.time()
        c0=comm.get().get_communication_stats()

        xo=self.multiheadMut(xo,self.bgin_M)
        xo=self.bgin_layer(xo)

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
        

        for i in range(self.config.num_hidden_layers-1):
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=self.multiheadMut(xo,self.M_mats[i])
            xo=self.d_ls[i](xo)

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

        xo=self.f_layer(xo)

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
