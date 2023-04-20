"""
======================================================================
ENCRYP_ENCDEC ---

Encoder Decoder Models with MERGE module.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 20 四月 2023
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

from utils import softmax_2RELU, softmax_2QUAD, activation_quad, activation_newGeLU, encrypt_tensor, softmax2RELU_2

from gpt import gptEmbeddings


class EncDecFlatten(nn.Module):
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
        self.embeddings.cuda(config.device)

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = config.hidden_size
        self.inter_d=config.intermediate_size
        self.LayerNorm = cnn.BatchNorm2d(config.hidden_size, eps=config.layer_norm_eps)

        if hasattr(config,"sequence_length"):
            self.msl=config.sequence_length
        else:
            self.msl=128

        self.enc_weight0_mats=[]
        self.enc_weight1_mats=[]
        self.enc_weight2_mats=[]
        self.enc_bias_mats=[]
        self.enc_bias1_mats=[]
        self.enc_M_mats=[]
        
        self.dec_self_weight0_mats=[]
        self.dec_self_weight1_mats=[]
        self.dec_self_M_mats=[]
        self.dec_self_bias_mats=[]

        self.dec_cross_weight0_mats=[]
        self.dec_cross_weight1_mats=[]
        self.dec_cross_M_mats=[]
        self.dec_cross_bias_mats=[]

        self.dec_weight2_mats=[]
        self.dec_bias1_mats=[]

        self.d=config.hidden_size
        self.I=config.intermediate_size
        self.enc_layers=self.config.encoder_layers
        self.dec_layers=self.config.decoder_layers

        ## ========================= precomputed matrix
        
        for num_layer in range(self.encoder_layers):
            self.enc_weight0_mats.append(torch.ones((self.d,
                                    self.d)).to(self.device))
            self.enc_weight1_mats.append(torch.ones((self.d,
                                    self.I)).to(self.device))
            self.enc_weight2_mats.append(torch.ones((self.I,
                                    self.d)).to(self.device))
            self.enc_bias_mats.append(torch.ones(self.I)\
                                  .to(self.device))
            self.enc_bias1_mats.append(torch.ones(self.d)\
                                  .to(self.device))
            self.enc_M_mats.append(torch.ones(self.num_attention_heads,
                            self.msl,self.msl).to(self.device))

        # encryption
        for i in range(self.config.num_hidden_layers):
            self.enc_weight0_mats[i]=crypten.cryptensor(self.enc_weight0_mats[i],
                                                src=0)
            self.enc_weight1_mats[i]=crypten.cryptensor(self.enc_weight1_mats[i],
                                                src=0)
            self.enc_weight2_mats[i]=crypten.cryptensor(self.enc_weight2_mats[i],
                                                src=0)
            self.enc_bias_mats[i]=crypten.cryptensor(self.enc_bias_mats[i],
                                                src=0)
            self.enc_M_mats[i]=crypten.cryptensor(self.enc_M_mats[i],
                                                src=0)
            
        for num_layer in range(self.decoder_layers):
            self.dec_self_weight0_mats.append(torch.ones((self.d,
                                    self.d)).to(self.device))
            self.dec_self_weight1_mats.append(torch.ones((self.d,
                                    self.I)).to(self.device))
            self.dec_self_M_mats.append(torch.ones(self.num_attention_heads,
                            self.msl,self.msl).to(self.device))
            self.dec_self_bias_mats.append(torch.ones(self.I)\
                                  .to(self.device))

            self.dec_cross_weight0_mats.append(torch.ones((self.d,
                                    self.d)).to(self.device))
            self.dec_cross_weight1_mats.append(torch.ones((self.d,
                                    self.I)).to(self.device))
            self.dec_cross_M_mats.append(torch.ones(self.num_attention_heads,
                            self.msl,self.msl).to(self.device))
            self.dec_cross_bias_mats.append(torch.ones(self.I)\
                                  .to(self.device))

            self.dec_weight2_mats.append(torch.ones((self.I,
                                    self.d)).to(self.device))
            self.dec_bias1_mats.append(torch.ones(self.d)\
                                  .to(self.device))

        # encryption
        for i in range(self.config.num_hidden_layers):
            self.dec_self_weight0_mats[i]=crypten.cryptensor(self.dec_self_weight0_mats[i],
                                                src=0)
            self.dec_self_weight1_mats[i]=crypten.cryptensor(self.dec_self_weight1_mats[i],
                                                src=0)
            self.dec_self_bias_mats[i]=crypten.cryptensor(self.dec_self_bias_mats[i],
                                                src=0)
            self.dec_self_M_mats[i]=crypten.cryptensor(self.dec_self_M_mats[i],
                                                src=0)

            self.dec_cross_weight0_mats[i]=crypten.cryptensor(self.dec_cross_weight0_mats[i],
                                                src=0)
            self.dec_cross_weight1_mats[i]=crypten.cryptensor(self.dec_cross_weight1_mats[i],
                                                src=0)
            self.dec_cross_bias_mats[i]=crypten.cryptensor(self.dec_cross_bias_mats[i],
                                                src=0)
            self.dec_cross_M_mats[i]=crypten.cryptensor(self.dec_cross_M_mats[i],
                                                src=0)

            self.dec_weight2_mats[i]=crypten.cryptensor(self.dec_weight2_mats[i],
                                                src=0)
            self.dec_bias1_mats[i]=crypten.cryptensor(self.bias1_mats[i],src=0)

        
        if config.hidden_act=="newGeLU":
            self.activation=activation_newGeLU()
        elif config.hidden_act=="relu":
            self.activation=nn.ReLU()
        elif config.hidden_act=="quad":
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
    def forward_enc(self.enc_idx):
        xo=self.embeddings(enc_idx)

        for i in range(self.encoder_layers):
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=self.multiheadMut(xo,self.enc_M_mats[i][:,:sl,:sl])
            xo=xo.matmul(self.enc_weight0_mats[i])
            xo=xo.matmul(self.enc_weight1_mats[i])
            xo=xo+self.enc_bias_mats[i]
            
            c1=comm.get().get_communication_stats()
            t1=time.time()

            self.timing["LinearTime"]+=(t1-t0)
            self.timing["LinearCommTime"]+=(c1['time']-c0['time'])
            self.timing["LinearCommByte"]+=(c1['bytes']-c0['bytes'])
            
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=self.activation(xo)

            c2=comm.get().get_communication_stats()
            t2=time.time()

            xo=xo.matmul(self.enc_weight2_mats[i])
            xo=xo+self.enc_bias1_mats[i]

            c1=comm.get().get_communication_stats()
            t1=time.time()
            xo=self.LayerNorm(xo)
            xo=self.LayerNorm(xo)

            self.timing["LinearTime"]+=(t1-t2)
            self.timing["LinearCommTime"]+=(c1['time']-c2['time'])
            self.timing["LinearCommByte"]+=(c1['bytes']-c2['bytes'])

            self.timing["ActivTime"]+=(t2-t0)
            self.timing["ActivCommTime"]+=(c2['time']-c0['time'])
            self.timing["ActivCommByte"]+=(c2['bytes']-c0['bytes'])
            enc_x=xo

    def forward(self,enc_x,dec_x):
        # print(x.shape)
        xo=self.embeddings(enc_x)

        for i in range(self.encoder_layers):
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=self.multiheadMut(xo,self.enc_M_mats[i][:,:sl,:sl])
            xo=xo.matmul(self.enc_weight0_mats[i])
            xo=xo.matmul(self.enc_weight1_mats[i])
            xo=xo+self.enc_bias_mats[i]
            
            c1=comm.get().get_communication_stats()
            t1=time.time()

            self.timing["LinearTime"]+=(t1-t0)
            self.timing["LinearCommTime"]+=(c1['time']-c0['time'])
            self.timing["LinearCommByte"]+=(c1['bytes']-c0['bytes'])
            
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=self.activation(xo)

            c2=comm.get().get_communication_stats()
            t2=time.time()

            xo=xo.matmul(self.enc_weight2_mats[i])
            xo=xo+self.enc_bias1_mats[i]

            c1=comm.get().get_communication_stats()
            t1=time.time()
            xo=self.LayerNorm(xo)
            xo=self.LayerNorm(xo)

            self.timing["LinearTime"]+=(t1-t2)
            self.timing["LinearCommTime"]+=(c1['time']-c2['time'])
            self.timing["LinearCommByte"]+=(c1['bytes']-c2['bytes'])

            self.timing["ActivTime"]+=(t2-t0)
            self.timing["ActivCommTime"]+=(c2['time']-c0['time'])
            self.timing["ActivCommByte"]+=(c2['bytes']-c0['bytes'])
            enc_x=xo

        for i in range(self.decoder_layers):
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=self.multiheadMut(xo,self.dec_self_M_mats[i][:,:sl,:sl])
            xo=xo.matmul(self.dec_self_weight0_mats[i])
            xo=xo.matmul(self.dec_self_weight1_mats[i])
            xo=xo+self.dec_self_bias_mats[i]

            ## noted that we use the enc_x here!
            xo=self.multiheadMut(enc_x,self.dec_cross_M_mats[i][:,:sl,:sl])
            xo=xo.matmul(self.dec_cross_weight0_mats[i])
            xo=xo.matmul(self.dec_cross_weight1_mats[i])
            xo=xo+self.dec_cross_bias_mats[i]
            
            c1=comm.get().get_communication_stats()
            t1=time.time()

            self.timing["LinearTime"]+=(t1-t0)
            self.timing["LinearCommTime"]+=(c1['time']-c0['time'])
            self.timing["LinearCommByte"]+=(c1['bytes']-c0['bytes'])
            
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=self.activation(xo)

            c2=comm.get().get_communication_stats()
            t2=time.time()

            xo=xo.matmul(self.dec_weight2_mats[i])
            xo=xo+self.dec_bias1_mats[i]

            c1=comm.get().get_communication_stats()
            t1=time.time()
            xo=self.LayerNorm(xo)
            xo=self.LayerNorm(xo)

            self.timing["LinearTime"]+=(t1-t2)
            self.timing["LinearCommTime"]+=(c1['time']-c2['time'])
            self.timing["LinearCommByte"]+=(c1['bytes']-c2['bytes'])

            self.timing["ActivTime"]+=(t2-t0)
            self.timing["ActivCommTime"]+=(c2['time']-c0['time'])
            self.timing["ActivCommByte"]+=(c2['bytes']-c0['bytes'])
            dec_x=xo

        return dec_x


    def forward2(self,enc_x,x):
        xo=self.embeddings(x)

        alist=[]
        sl=xo.shape[1]

        for i in range(self.decoder_layers):
            alist.append(xo)
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=self.multiheadMut(xo,self.dec_self_M_mats[i][:,:sl,:sl])
            xo=xo.matmul(self.dec_self_weight0_mats[i])
            xo=xo.matmul(self.dec_self_weight1_mats[i])
            xo=xo+self.dec_self_bias_mats[i]

            ## noted that we use the enc_x here!
            xo=self.multiheadMut(enc_x,self.dec_cross_M_mats[i][:,:sl,:sl])
            xo=xo.matmul(self.dec_cross_weight0_mats[i])
            xo=xo.matmul(self.dec_cross_weight1_mats[i])
            xo=xo+self.dec_cross_bias_mats[i]
            
            c1=comm.get().get_communication_stats()
            t1=time.time()

            self.timing["LinearTime"]+=(t1-t0)
            self.timing["LinearCommTime"]+=(c1['time']-c0['time'])
            self.timing["LinearCommByte"]+=(c1['bytes']-c0['bytes'])
            
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=self.activation(xo)

            c2=comm.get().get_communication_stats()
            t2=time.time()

            xo=xo.matmul(self.dec_weight2_mats[i])
            xo=xo+self.dec_bias1_mats[i]

            c1=comm.get().get_communication_stats()
            t1=time.time()
            xo=self.LayerNorm(xo)
            xo=self.LayerNorm(xo)

            self.timing["LinearTime"]+=(t1-t2)
            self.timing["LinearCommTime"]+=(c1['time']-c2['time'])
            self.timing["LinearCommByte"]+=(c1['bytes']-c2['bytes'])

            self.timing["ActivTime"]+=(t2-t0)
            self.timing["ActivCommTime"]+=(c2['time']-c0['time'])
            self.timing["ActivCommByte"]+=(c2['bytes']-c0['bytes'])
            dec_x=xo

        ## final transform
        alist.append(xo)

        return xo,alist

    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0

    def one_step(self,enc_idx,new_idx,past_states):

        new_idx=new_idx.unsqueeze(1)
        
        enc_x=self.embeddings(enc_idx) # shape should be: bs,1,d
        xo=self.embeddings(new_idx) # shape should be: bs,1,d

        return self.feature_onestep(enc_x,xo,past_states)

    # @pysnooper.snoop(watch=('xo.shape',))
    def feature_onestep(self,enc_x,new_feature,past_states):
        """
        shape of new_feature: bs,d
        """
        if len(past_states[0])==0:
            num_past_token=0 # i.e. sequence length
        else:
            num_past_token=past_states[0].shape[1] # i.e. sequence length

        xo=new_feature

        L=self.config.decoder_layers
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
            
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=self.multiheadMut(past_states[i],
                                 self.dec_self_M_mats[i][:,
                                    :num_past_token+1,
                                    num_past_token:num_past_token+1])
            xo=xo.matmul(self.dec_self_weight0_mats[i])
            xo=xo.matmul(self.dec_self_weight1_mats[i])
            xo=xo+self.dec_self_bias_mats[i]

            ## to do: need to modify this part.
            xo=self.multiheadMut(enc_x,
                                 self.dec_cross_M_mats[i][:,
                                    :num_past_token+1,
                                    num_past_token:num_past_token+1])

            xo=xo.matmul(self.dec_cross_weight0_mats[i])
            xo=xo.matmul(self.dec_cross_weight1_mats[i])
            xo=xo+self.dec_cross_bias_mats[i]

            c1=comm.get().get_communication_stats()
            t1=time.time()

            self.timing["LinearTime"]+=(t1-t0)
            self.timing["LinearCommTime"]+=(c1['time']-c0['time'])
            self.timing["LinearCommByte"]+=(c1['bytes']-c0['bytes'])

            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=self.activation(xo)

            c2=comm.get().get_communication_stats()
            t2=time.time()

            xo=xo.matmul(self.dec_weight2_mats[i])
            xo=xo+self.dec_bias1_mats[i]

            c1=comm.get().get_communication_stats()
            t1=time.time()

            
            self.timing["LinearTime"]+=(t1-t2)
            self.timing["LinearCommTime"]+=(c1['time']-c2['time'])
            self.timing["LinearCommByte"]+=(c1['bytes']-c2['bytes'])

            self.timing["ActivTime"]+=(t2-t0)
            self.timing["ActivCommTime"]+=(c2['time']-c0['time'])
            self.timing["ActivCommByte"]+=(c2['bytes']-c0['bytes'])

        past_states[i+1]=past_states[i+1].\
            cat([past_states[i+1],xo], 1)

        return xo
        
            
    def generate_vanilla(self,enc_idx, idx):
        """
        vainilla conditional generation
           idx: input one-hot sparce tensor.
        """
        past_list = [[] for _ in range(self.config.num_hidden_layers)]
        generation_stage = False
        enc_x=self.forward_enc(enc_idx)
        _,past_list=self.forward2(enc_x,idx[:,:-1,:])

        prog=tqdm(total=self.config.gen_len)
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

            feature = self.one_step(enc_x,idx_cond[:,-1,:], past_list)
            # print(past_list[0].shape)
            
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

    def generate_ourmethod(self,enc_idx,idx):
        """fast forward inference proposed by our method."""
        past_features = [[] for _ in range(self.config.num_hidden_layers+1)]
        generation_stage = False

        bs,sql,v=idx.shape
        if sql<1: # empty
            idx=self.cat([idx,self.bos_one_hot])

        feature=self.embeddings(idx[:,-1:,:])
        
        enc_x=self.forward_enc(enc_idx)
        _,past_features=self.forward2(enc_x,idx[:,:-1,:])

        prog=tqdm(total=self.config.gen_len)
        
        for _ in range(self.config.gen_len):
            prog.update(1)
            # try:
            #     print(past_features[0].shape)
            # except:
            #     print(past_features[0])

            # truncation
            idx_cond = idx if\
                idx.size(1) <= self.config.max_position_embeddings\
                else idx[:, -self.config.max_position_embeddings:,:]
            # feature,past_features=self.feature_onestep(feature,
                                                       # past_features)
            ## note:
            # HERE WE USE THE REFERNCE-TRANSMIT INSEATED OF VALUE-TRANSMIT.
            # 此处使用引用传递而非值传递
            feature=self.feature_onestep(enc_x,feature,
                                         past_features)

        t0=time.time()
        c0=comm.get().get_communication_stats()
        all_logits=self.lm_head(past_features[-1])
        c1=comm.get().get_communication_stats()
        t1=time.time()
        self.timing["GenerationTime"]+=(t1-t0)
        self.timing["GenerationCommTime"]+=(c1['time']-c0['time'])
        self.timing["GenerationCommByte"]+=(c1['bytes']-c0['bytes'])
        
        return all_logits
