"""
======================================================================
ENCODERDECODER_VANILLA ---

Vanilla Transformer Module of Encoder-Decoder Generations.

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

import time
import math

import torch
import torch.nn.functional as F

import crypten
import crypten.nn as cnn
import crypten.communicator as comm
from crypten.common.functions import maximum

from tqdm import tqdm
from utils import softmax_2RELU, softmax_2QUAD, activation_quad, activation_newGeLU, encrypt_tensor, softmax2RELU_2
from models import BertLayer
from gpt import *

class gptDecLayer(cnn.Module):
    def __init__(self, config, timing):
        super(gptDecLayer, self).__init__()
        self.config = config
        self.attention = gptAttention(config, timing)
        self.cattention = gptCrossAttention(config, timing)
        self.intermediate = gptIntermediate(config, timing)
        self.output = gptOutput(config, timing)
        self.config = config
        self.timing = timing
 
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0
 
    def forward(self,enc_x, hidden_states, past):
        #attention_output, past = self.attention(hidden_states, past)
        #print("debug copy before: ", past)
        attention_output = self.attention(hidden_states, past)
        attention_output = self.cattention(enc_x,hidden_states, past)
        #print("debug copy after: ", past)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output#, past

class gptCrossAttention(cnn.Module):
    def __init__(self, config, timing):

        super(gptCrossAttention, self).__init__()
        self.self = gptOnlyCrossAttention(config, timing)
        self.output = gptSelfOutput(config, timing)
    
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0
    
    def forward(self,enc_x, hidden_states, past):
        #self_output, past = self.self(hidden_states, past)
        self_output = self.self(enc_x,hidden_states, past)
        attention_output = self.output(self_output, hidden_states)
        return attention_output#, past

class gptOnlyCrossAttention(cnn.Module):
    def __init__(self, config, timing):
        super(gptOnlyCrossAttention, self).__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = self.hidden_size // self.num_attention_heads

        self.query = cnn.Linear(self.hidden_size, self.hidden_size)
        self.key = cnn.Linear(self.hidden_size, self.hidden_size)
        self.value = cnn.Linear(self.hidden_size, self.hidden_size)

        self.cat = cnn.Concat(dimension=-2)
        self.dropout = cnn.Dropout(config.attention_probs_dropout_prob)
        if config.softmax_act == "softmax":
            self.smax = cnn.Softmax(dim=-1)
        elif config.softmax_act == "softmax_2RELU":
            self.smax = softmax_2RELU(dim=-1)
        elif config.softmax_act == "softmax2RELU_2":
            self.smax = softmax2RELU_2(dim=-1)
        elif config.softmax_act == "softmax_2QUAD":
            self.norm = cnn.BatchNorm2d(config.hidden_size, eps=config.layer_norm_eps)
            self.smax = softmax_2QUAD(self.norm, dim=-1)
        else:
           raise ValueError(f"softmax type {config.softmax_act} not implemented.")
        self.timing = timing
    
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,enc_x, hidden_states, past):
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(enc_x))
        print("key shape:", key_layer.shape)
        value_layer = self.transpose_for_scores(self.value(enc_x))
        
        if len(past) != 0:
            past_key, past_value = past
            print("cat debug: ", past_key.shape, key_layer.shape )
            key_layer = self.cat([past_key, key_layer])
            value_layer = self.cat([past_value, value_layer])
            past[0] = key_layer
            past[1] = value_layer
        else:
            # update past
            past.append(key_layer)
            past.append(value_layer)        
           
        attention_scores = query_layer.matmul(key_layer.transpose(-1, -2))
        #print(attention_scores.shape)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # TODO: implement mask
        # attention_scores = attention_scores.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["LinearTime"] += (t1 - t0)
        self.timing["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["LinearCommByte"] += (comm1["bytes"] - comm0["bytes"])
        
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        #print("smax operands: ", attention_scores.shape)
        attention_probs = self.smax(attention_scores)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["SoftmaxTime"] += (t1 - t0)
        self.timing["SoftmaxCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["SoftmaxCommByte"] += (comm1["bytes"] - comm0["bytes"])

        attention_probs = self.dropout(attention_probs)
        # print(f"Attention shape: {attention_probs.shape}")
        # print(f"Value shape: {value_layer.shape}")
        
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        context_layer = attention_probs.matmul(value_layer)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["LinearTime"] += (t1 - t0)
        self.timing["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["LinearCommByte"] += (comm1["bytes"] - comm0["bytes"])

        # print(f"context shape{context_layer.shape}")
        context_layer = context_layer.permute(0, 2, 1, 3)#.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        #print("debug shapes after attention: ", context_layer.shape, key_layer.shape, value_layer.shape)        
        return context_layer#, (key_layer, value_layer)


class EncdrDecdr(cnn.Module):
    def __init__(self,config,timing):
        super(EncdrDecdr,self).__init__()
        self.config=config

        # shared embeddings
        self.embeddings=gptEmbeddings(config,timing)
        self.embeddings.cuda(config.device)
        self.encoder = cnn.ModuleList([BertLayer(config,timing) for\
                                       _ in range(config.encoder_layers)])
        self.decoder = cnn.ModuleList([gptDecLayer(config,timing) for\
                                       _ in range(config.decoder_layers)])

        self.lm_head=cnn.Linear(config.hidden_size,config.vocab_size)
        self.smax=cnn.Softmax(dim=-1)
        self.cat=cnn.Concat(dimension=1)
        self.timing=timing
        
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k]=0

    # def forward(self,input_ids,dec_input_ids,past_list):
    def forward(self,enc_x,dec_input_ids,past_list):
        ## 1. get embeddings
        # enc_x=self.embeddings(input_ids)

        ## 2. get the hidden states of transformer encoder.
        for l_id, layer in enumerate(self.encoder):
            enc_x=layer(enc_x)

        ## 3. get the hidden states of transformer decoder.
        dec_x=self.embeddings(dec_input_ids)
        for dl_id,layer in enumerate(self.decoder):
            dec_x=layer(enc_x,dec_x,past_list[dl_id])
        
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        output = self.lm_head(dec_x)
        comm1 = comm.get().get_communication_stats()
        t1 = time.time()
            
        self.timing["GenerateOtherTime"] += (t1-t0)
        self.timing["GenerateOtherCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["GenerateOtherCommByte"] += (comm1["bytes"] - comm0["bytes"])
        return output#, past


    def forward_nohead(self,enc_x,dec_input_ids,past_list):
        ## 3. get the hidden states of transformer decoder.
        dec_x=self.embeddings(dec_input_ids)
        for dl_id,layer in enumerate(self.decoder):
            dec_x=layer(enc_x,dec_x,past_list[dl_id])
        return dec_x

    def forward_enc(self,input_ids,):
        ## 1. get embeddings
        enc_x=self.embeddings(input_ids)

        ## 2. get the hidden states of transformer encoder.
        for l_id, layer in enumerate(self.encoder):
            enc_x=layer(enc_x)
        return enc_x

    def forward_noembed(self,enc_x,dec_feature,past_list):
        # dec_x=feature
        dec_x=dec_feature

        ## 3. get the hidden states of transformer decoder.
        for dl_id,layer in enumerate(self.decoder):
            dec_x=layer(enc_x,dec_x,past_list[dl_id])
        return dec_x

    def feature_onestep(self,enc_x,new_feature,past_states):
        """
        shape of new_feature: bs,d
        """

        xo=new_feature
        xo=self.forward_noembed(enc_x,xo,past_states)
        return xo

    def generate_ourmethod(self,enc_idx, idx, max_new_tokens,):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,s,v)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        generation_time = {}
        past_list = [[] for _ in range(self.config.decoder_layers)]
        generation_stage = False

        enc_x=self.forward_enc(enc_idx)

        b, sql, _ = idx.shape
        if sql<1: # empty
            idx=self.cat([idx,self.bos_one_hot])
        feature=self.embeddings(idx[:,-1:,:])
        output=self.forward_nohead(enc_x,idx[:,:-1,:],past_list)
        past_features=past_list

        prog=tqdm(total=self.config.sequence_length)

        for _ in range(self.config.gen_len):
            prog.update(1)
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.max_position_embeddings else idx[:, -self.config.max_position_embeddings:,:]
            feature = self.feature_onestep(enc_x,feature,
                                          past_features)
            
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        logits=self.lm_head(feature)
        comm1 = comm.get().get_communication_stats()
        t1 = time.time()
        self.timing["GenerateTime"] += (t1-t0)
        self.timing["GenerateCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["GenerateCommByte"] += (comm1["bytes"] - comm0["bytes"])
        return idx

    def generate(self, enc_idx, idx, max_new_tokens,
                 temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,s,v)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        generation_time = {}
        past_list = [[] for _ in range(self.config.decoder_layers)]
        generation_stage = False
        enc_x=self.forward_enc(enc_idx)
        for token_id in range(max_new_tokens):
            b, s, _ = idx.shape
            time_s = time.time()
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.max_position_embeddings else idx[:, -self.config.max_position_embeddings:,:]
            # forward the model to get the logits for the index in the sequence
            #print(idx_cond.shape)
            if not generation_stage:
                logits = self(enc_x,idx_cond, past_list)
                generation_stage = True
            else:
                logits = self(enc_x,idx_cond[:, -1:, :], past_list)
            #print("logit shape: ", logits.shape)
            # pluck the logits at the final step and scale by desired temperature
            t0 = time.time()
            comm0 = comm.get().get_communication_stats()
            logits = logits[:, -1:, :] / temperature
            #print("logits shape: ", logits.shape)
            # optionally crop the logits to only the top k options
            #if top_k is not None:
            #    v, _ = torch.topk(logits, top_k)
            #    logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = self.smax(logits)
            # either sample from the distribution or take the most likely element
            #if do_sample:
            #    idx_next = torch.multinomial(probs, num_samples=1)
            #else:
            #print("prob size: ", probs.shape)
            idx_next = maximum.argmax(probs, dim=-1)
            #print("next idx:", idx_next.shape)
            # append sampled index to the running sequence and continue
            #idx_next = F.one_hot(idx_next, self.config.vocab_size).cuda()
            #idx_next = encrypt_tensor(idx_next)
            #print("pre-cat size: ",idx.shape, idx_next.shape)
            idx = self.cat([idx, idx_next])
            comm1 = comm.get().get_communication_stats()
            t1 = time.time()
            time_e = time.time()
            generation_time.update({(b, s): time_e - time_s})
            self.timing["GenerateOtherTime"] += (t1-t0)
            self.timing["GenerateOtherCommTime"] += (comm1["time"] - comm0["time"])
            self.timing["GenerateOtherCommByte"] += (comm1["bytes"] - comm0["bytes"])
            print(generation_time)
        return idx
