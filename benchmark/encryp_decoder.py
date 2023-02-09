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
        # print(config.device)
        self.device=torch.device(config.device)
        device=self.device

        self.bos_one_hot=F.one_hot(torch.randint(low=0,
            high=config.vocab_size,
            size=(1,)),
            config.vocab_size).float().to(self.device)
        self.bos_one_hot=crypten.cryptensor(self.bos_one_hot)

        self.embeddings=gptEmbeddings(config,timing)
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
        self.d=config.hidden_size
        self.I=config.intermediate_size
        ## ========================= precomputed matrix
        self.bgin_layer=nn.Linear(config.hidden_size,self.I)
        self.bgin_w=torch.ones((self.I,config.hidden_size)).to(device).T
        self.bgin_b=torch.ones(self.I).to(device)
        self.bgin_M=torch.ones(self.num_attention_heads,
                               self.msl,self.msl).to(device)
        self.bgin_w=crypten.cryptensor(self.bgin_w,src=0)
        self.bgin_b=crypten.cryptensor(self.bgin_b,src=0)
        self.bgin_M=crypten.cryptensor(self.bgin_M,src=0)
        
        for num_layer in range(self.config.num_hidden_layers-1):
            self.weight1_mats.append(torch.ones((self.d,
                                    self.I)).to(self.device).T)
            self.weight2_mats.append(torch.ones((self.I,
                                    self.d)).to(self.device).T)
            self.bias_mats.append(torch.ones(self.I)\
                                  .to(self.device))
            self.M_mats.append(torch.ones(self.num_attention_heads,
                            self.msl,self.msl).to(self.device))
        self.last_w=torch.ones((self.d,self.I)).to(self.device).T
        self.last_b=torch.ones((config.hidden_size)).to(self.device)

        # incryption
        for i in range(self.config.num_hidden_layers-1):
            self.weight1_mats[i]=crypten.cryptensor(self.weight1_mats[i],
                                                src=0)
            self.weight2_mats[i]=crypten.cryptensor(self.weight2_mats[i],
                                                src=0)
            self.bias_mats[i]=crypten.cryptensor(self.bias_mats[i],
                                                src=0)
            self.M_mats[i]=crypten.cryptensor(self.M_mats[i],
                                                src=0)
        
        self.activation=activation_quad()
        # self.activation=nn.ReLU()

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
    def forward2(self,x):
        xo=self.embeddings(x)

        alist=[xo]
        sl=xo.shape[1]

        t0=time.time()
        c0=comm.get().get_communication_stats()

        xo=xo.matmul(self.bgin_w)
        xo=self.multiheadMut(xo,self.bgin_M[:,:sl,:sl])
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
            alist.append(xo)
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=self.multiheadMut(xo,self.M_mats[i][:,:sl,:sl])
            xo=xo.matmul(self.weight1_mats[i])
            xo=xo.matmul(self.weight2_mats[i])
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

        alist.append(xo)
        ## final transform
        t0=time.time()
        c0=comm.get().get_communication_stats()
        xo=xo.matmul(self.last_w)+self.last_b
        c1=comm.get().get_communication_stats()
        t1=time.time()
        self.timing["LinearTime"]+=(t1-t0)
        self.timing["LinearCommTime"]+=(c1['time']-c0['time'])
        self.timing["LinearCommByte"]+=(c1['bytes']-c0['bytes'])
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
        if len(past_states[0])==0:
            past_states[0]=xo
        else:
            past_states[0]=past_states[0].\
                cat([past_states[0],xo], 1)

        t0=time.time()
        c0=comm.get().get_communication_stats()

        xo=self.multiheadMut(past_states[0],
                                self.bgin_M[:,
                :num_past_token+1,-1].unsqueeze(2))
        xo=xo.matmul(self.bgin_w)
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
        
        
        L=self.config.num_hidden_layers-1
        for i in range(L):
            # the outputs of previous layer
            # print("xo type and shape",type(xo),xo.shape)
            # print("shape before cat: ",past_states[i].shape)
            if len(past_states[i+1])==0:
                past_states[i+1]=xo
            else:
                past_states[i+1]=past_states[i+1].\
                    cat([past_states[i+1],xo], 1)
            # print("shape after cat: ",past_states[i].shape)
            
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=self.multiheadMut(past_states[i+1],
                                 self.M_mats[i][:,
                    :num_past_token+1,-1].unsqueeze(2))

            xo=xo.matmul(self.weight1_mats[i])
            xo=xo.matmul(self.weight2_mats[i])
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
        
        past_states[i+2]=past_states[i+2].\
            cat([past_states[i+2],xo], 1)

        t0=time.time()
        c0=comm.get().get_communication_stats()
        xo=xo.matmul(self.last_w)+self.last_b
        c1=comm.get().get_communication_stats()
        t1=time.time()
        self.timing["LinearTime"]+=(t1-t0)
        self.timing["LinearCommTime"]+=(c1['time']-c0['time'])
        self.timing["LinearCommByte"]+=(c1['bytes']-c0['bytes'])

        past_states[i+3]=past_states[i+3].\
            cat([past_states[i+3],xo], 1)

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
            
            t0=time.time()
            c0=comm.get().get_communication_stats()
            logits=self.lm_head(feature)
            c1=comm.get().get_communication_stats()
            t1=time.time()
            self.timing["LinearTime"]+=(t1-t0)
            self.timing["LinearCommTime"]+=(c1['time']-c0['time'])
            self.timing["LinearCommByte"]+=(c1['bytes']-c0['bytes'])

            probs = self.smax(logits)
            idx_next = maximum.argmax(probs, dim=-1)
            c2=comm.get().get_communication_stats()
            t2=time.time()
            self.timing["SMAMTime"]+=(t2-t1)
            self.timing["SMAMCommTime"]+=(c2['time']-c1['time'])
            self.timing["SMAMCommByte"]+=(c2['bytes']-c1['bytes'])

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

        t0=time.time()
        c0=comm.get().get_communication_stats()
        all_logits=self.lm_head(past_features[-1])
        c1=comm.get().get_communication_stats()
        t1=time.time()
        self.timing["LinearTime"]+=(t1-t0)
        self.timing["LinearCommTime"]+=(c1['time']-c0['time'])
        self.timing["LinearCommByte"]+=(c1['bytes']-c0['bytes'])
        
        return all_logits

