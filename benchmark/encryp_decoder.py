"""
======================================================================
ENCRYP_DECODER ---

Encryption transformer decoder for private inference.

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
from copy import deepcopy
from tqdm import tqdm

import time
import math

import pysnooper
import torch
import torch.nn.functional as F

import crypten
import crypten.nn as cnn
import crypten.nn as nn
import crypten.communicator as comm
from crypten.common.functions import maximum

from utils import softmax_2RELU, softmax_2QUAD, activation_quad, activation_newGeLU, encrypt_tensor

from gpt import gptEmbeddings

class GPTBaseFlatten(nn.Module):
    def __init__(self, config, timing):
        super(GPTBaseFlatten,self).__init__()
        self.config=config
        self.timing=timing
        # self.config.num_labels=2
        self.device=torch.device("cuda:0")

        self.bos_one_hot=F.one_hot(torch.randint(low=0,
            high=config.vocab_size,
            size=(1,)),
            config.vocab_size).float().cuda()

        self.embeddings=gptEmbeddings(config,timing)

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

        self.lm_head=cnn.Linear(config.hidden_size,
                                config.vocab_size,bias=False)
        self.smax = cnn.Softmax(dim=-1)
        self.cat = cnn.Concat(dimension=1)

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

        for i in range(self.config.num_hidden_layers):
            xo=xo.matmul(self.weight_mats[i].T)
            xo=self.multiheadMut(xo,self.M_mats[i])
            xo=xo+self.bias_mats[i]
            xo=self.activation(xo)

        ## final transform
        xo=xo.matmul(self.last_w.T)+self.last_b

        return xo
    def forward2(self,x):
        # print(x.shape)
        xo=self.embeddings(x)
        alist=[]
        sl=xo.shape[1]

        for i in range(self.config.num_hidden_layers):
            alist.append(xo)
            # print(xo.shape)
            # print(self.M_mats[i][:,:sl,:sl].shape)
            xo=self.multiheadMut(xo,self.M_mats[i][:,:sl,:sl])
            xo=xo.matmul(self.weight_mats[i].T)
            xo=xo+self.bias_mats[i]
            xo=self.activation(xo)

        alist.append(xo)
        ## final transform
        xo=xo.matmul(self.last_w.T)+self.last_b
        alist.append(xo)

        return xo,alist

    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0

    def one_step(self,new_idx,past_states):

        new_idx=new_idx.unsqueeze(1)
        xo=self.embeddings(new_idx) # shape should be: bs,1,d
        return self.feature_onestep(xo,past_states)

    # @pysnooper.snoop(watch=('xo.shape',))
    def feature_onestep(self,new_feature,past_states):
        """
        shape of new_feature: bs,d
        """
        if len(past_states[0])==0:
            num_past_token=0 # i.e. sequence length
        else:
            num_past_token=past_states[0].shape[1] # i.e. sequence length

        xo=new_feature
        L=self.config.num_hidden_layers
        for i in range(L):
            # the outputs of previous layer
            # print("xo type and shape",type(xo),xo.shape)
            # print("shape before cat: ",past_states[i].shape)
            if len(past_states[i])==0:
                past_states[i]=xo
            else:
                past_states[i]=past_states[i].\
                    cat([past_states[i],xo], 1)
            # print("shape after cat: ",past_states[i].shape)
            
            xo=self.multiheadMut(past_states[i],
                                 self.M_mats[i][:,
                    :num_past_token+1,-1].unsqueeze(2))

            xo=xo.matmul(self.weight_mats[i].T)
            xo=xo+self.bias_mats[i]
            xo=self.activation(xo)
        
        past_states[i+1]=past_states[i+1].\
            cat([past_states[i+1],xo], 1)
        xo=xo.matmul(self.last_w.T)+self.last_b
        past_states[i+2]=past_states[i+2].\
            cat([past_states[i+1],xo], 1)

        # print("--------")
        # print(xo)
        # print(xo.shape)

        return xo,past_states
        
            
    def generate_vanilla(self, idx):
        """
        vainilla conditional generation
           idx: input one-hot sparce tensor.
        """
        past_list = [[] for _ in range(self.config.num_hidden_layers)]
        generation_stage = False
        _,past_list=self.forward2(idx[:,:-1,:])

        prog=tqdm(total=self.config.sequence_length-\
                  self.config.prefix_length)
        while True:
            prog.update(1)
            b, s, _ = idx.shape
            if s>=self.config.sequence_length:
                break
            time_s = time.time()
            # truncation
            idx_cond = idx if\
                idx.size(1) <= self.config.max_position_embeddings\
                else idx[:, -self.config.max_position_embeddings:,:]

            feature,past_list = self.one_step(idx_cond[:,-1,:], past_list)
            logits=self.lm_head(feature)

            probs = self.smax(logits)
            idx_next = maximum.argmax(probs, dim=-1)
            idx = self.cat([idx, idx_next])
        return idx

    def generate_ourmethod(self,idx):
        """fast forward inference proposed by our method."""
        past_features = [[] for _ in range(self.config.num_hidden_layers+2)]
        generation_stage = False

        bs,sql,v=idx.shape
        if sql<1: # empty
            idx=self.cat([idx,self.bos_one_hot])

        feature=self.embeddings(idx[:,-1,:].unsqueeze(1))
        _,past_features=self.forward2(idx[:,:-1,:])

        prog=tqdm(total=self.config.sequence_length-\
                  self.config.prefix_length)
        while True:
            prog.update(1)
            s=past_features[0].shape[1]+1
            if s>=self.config.sequence_length:
                break
            # truncation
            idx_cond = idx if\
                idx.size(1) <= self.config.max_position_embeddings\
                else idx[:, -self.config.max_position_embeddings:,:]
            feature,past_features=self.feature_onestep(feature,
                                                       past_features)
        return idx

