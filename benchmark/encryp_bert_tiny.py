"""
======================================================================
ENCRYP_BERT_TINY ---

Encryption for precomputed BERT-tiny models

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

class BertTinyFlatten(nn.Module):
    def __init__(self, config, timing):
        super(BertTinyFlatten,self).__init__()
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

        # ## vanilla version
        # self.M = nn.Parameter(torch.zeros(self.num_attention_heads,
        #                 self.msl,
        #                 self.msl))
        # self.Wvalue = nn.Linear(config.hidden_size, self.all_head_size)
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # self.activation=XXXX
        # self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        ## Linear Version
        self.M = torch.zeros(self.num_attention_heads,
                        self.msl,
                        self.msl)
        self.Wvalue = nn.Linear(config.hidden_size, self.all_head_size)
        self.dense0 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNormweight = torch.zeros(config.hidden_size)
        self.LayerNormbias = torch.zeros(config.hidden_size)
        self.denseFF0 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation=nn.ReLU()
        self.denseFF1 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNormFFweight = torch.zeros(config.hidden_size)
        self.LayerNormFFbias = torch.zeros(config.hidden_size)

        ## Linear Version
        self.M1 = torch.zeros(self.num_attention_heads,
                        self.msl,
                        self.msl)
        self.Wvalue1 = nn.Linear(config.hidden_size, self.all_head_size)
        self.dense01 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNormweight1 = torch.zeros(config.hidden_size)
        self.LayerNormbias1 = torch.zeros(config.hidden_size)
        self.denseFF01 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation1=nn.ReLU()
        self.denseFF11 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNormFFweight1 = torch.zeros(config.hidden_size)
        self.LayerNormFFbias1 = torch.zeros(config.hidden_size)

        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        ## ========================= precomputed matrix
        print(type(self.Wvalue.weight),type(self.dense0.weight))
        self.init_d_flatten=torch.matmul(self.Wvalue.weight,self.dense0.weight)
        self.init_d_flatten=torch.tensor(self.init_d_flatten)*self.LayerNormweight
        self.init_d_flatten=torch.matmul(self.denseFF0.weight,
                                         self.init_d_flatten)
        self.init_d_bias_flatten=mm(self.denseFF0.weight,
                                              self.LayerNormbias) + \
            mm(self.denseFF0.weight,self.dense0.bias) +\
            self.denseFF0.bias

        self.init_M=self.M

        self.inter0_d_flatten=mm(self.denseFF1.weight.T,
                                 self.Wvalue1.weight)\
            *self.LayerNormFFweight

        self.inter0_d_flatten=torch.mm(self.inter0_d_flatten,
                                           self.dense01.weight,
                                           ).T

        self.inter0_d_flatten=torch.mul(self.inter0_d_flatten,
                                        self.LayerNormweight1.unsqueeze(1)\
                                        .expand(
            self.all_head_size,self.inter_d))
        self.inter0_d_flatten=torch.matmul(self.denseFF01.weight,
                                           self.inter0_d_flatten,
                                           )

        inter0_b_p1=mm(self.denseFF01.weight,mm(mm(self.denseFF1.bias,
                          self.Wvalue1.weight),
                          self.dense01.weight)*self.LayerNormFFweight1
                       )
        inter0_bp2=mm(self.denseFF01.weight,mm(self.Wvalue1.bias,
                          self.dense01.weight)*self.LayerNormFFweight1,
                       )
        inter0_bp3=mm(self.denseFF01.weight,self.dense01.bias\
                      *self.LayerNormFFweight1)
        inter0_bp4=self.denseFF01.bias
        inter0_b_p5=mm(self.LayerNormFFbias,
            torch.mm(self.Wvalue1.weight,
                     self.dense01.weight))*self.LayerNormFFweight1
        inter0_b_p5=mm(inter0_b_p5,self.denseFF01.weight.T)
        inter0_b_p6=mm(self.LayerNormbias1,self.denseFF01.weight.T)

        self.inter0_bias=inter0_b_p1+inter0_bp2+inter0_bp3+inter0_bp4+\
            inter0_b_p5+inter0_b_p6
        self.inter0_M=self.M1

        # print(self.denseFF11.weight.shape,self.LayerNormFFweight1.shape)
        self.final_d=self.denseFF11.weight*self.LayerNormFFweight1\
                                    .unsqueeze(1).expand(self.all_head_size,
                                                         self.inter_d)
        
        self.final_b=self.denseFF11.bias*self.LayerNormFFweight1.\
            unsqueeze(0).expand(-1,self.all_head_size)\
        +self.LayerNormFFbias1

        self.activation=activation_quad()

        # self.init_d_flatten=nn.Parameter(self.init_d_flatten)
        self.init_d_flatten,self.init_M,self.inter0_d_flatten,self.inter0_bias,self.final_d,self.final_b=self.init_d_flatten.to(self.device),self.init_M.to(self.device),self.inter0_d_flatten.to(self.device),self.inter0_bias.to(self.device),self.final_d.to(self.device),self.final_b.to(self.device)
        self.init_d_bias_flatten=self.init_d_bias_flatten.to(self.device)
        self.inter0_M=self.inter0_M.to(self.device)

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

        ## init transform
        xo=xo.matmul(self.init_d_flatten.T)
        xo=self.multiheadMut(xo,self.init_M)
        xo=xo+self.init_d_bias_flatten
        xo=self.activation(xo)

        ## internal0 transform
        xo=xo.matmul(self.inter0_d_flatten.T)
        xo=self.multiheadMut(xo,self.inter0_M)
        xo=xo+self.inter0_bias
        xo=self.activation(xo)

        ## final transform
        xo=xo.matmul(self.final_d.T)+self.final_b

        # logits=self.classifier(xo[:,0,:])
        # return logits 
        return xo

    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")
