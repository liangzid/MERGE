"""
======================================================================
PROFILE_ENCDRDECDR --- 

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

import sys
import os
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import AutoConfig, BertForSequenceClassificationWrapper

import crypten
import crypten.communicator as comm
from crypten.config import cfg
from utils import encrypt_tensor, encrypt_model

from encoderDecoder_vanilla import EncdrDecdr as gpt
from encryp_encdec import EncDecFlatten as GPTBaseFlatten

# from encryp_decoder import GPTBaseFlatten
# from encryp_decoder_nosimLN import GPTBaseFlatten

# 2PC setting
rank = sys.argv[1]
device=str(sys.argv[2])
if device=="-1" or device=="cpu":
   device="cpu"
else:
   device=f"cuda:{device}"
h=int(sys.argv[3])
d=int(sys.argv[4])
msl=int(sys.argv[5])
prefix_len=int(sys.argv[6])
num_head=int(sys.argv[7])
method=str(sys.argv[8])
gen_type=str(sys.argv[9])
port=str(sys.argv[10])


os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(2)
# os.environ["MASTER_ADDR"] = "219.245.186.45"
os.environ["MASTER_ADDR"] = "219.245.186.48"
os.environ["MASTER_PORT"] = port
os.environ["RENDEZVOUS"] = "env://"


# Inference arguments
class config():
    def __init__(self):
        self.batch_size = 1
        self.encoder_layers=int(h//2)
        self.decoder_layers=int(h//2)
        self.hidden_size = d
        self.intermediate_size = self.hidden_size * 4
        self.sequence_length=msl
        self.max_position_embeddings = self.sequence_length
        if method=="vanillaGPT":
            self.hidden_act="newGeLU"
            self.softmax_act="softmax"
        elif method=="thex":
            self.hidden_act="relu"
            self.softmax_act="softmax2RELU_2"
        elif method=="onlyMM":
            self.hidden_act="newGeLU"
            self.softmax_act="softmax"
        elif method=="onlyER":
            self.hidden_act="newGeLU"
            self.softmax_act="softmax"
        elif method=="mpcformer_sfrelu":
            self.hidden_act="quad"
            self.softmax_act="softmax_2RELU"
        elif method=="mpcformer_sfquad":
            self.hidden_act="quad"
            self.softmax_act="softmax_2QUAD"
        else:
            #self.hidden_act = "newGeLU"
            #self.softmax_act = "softmax"
            self.hidden_act = "quad"
            self.softmax_act = "softmax_2QUAD"
        self.layer_norm_eps = 1e-12
        self.num_attention_heads = num_head
        self.vocab_size = 50257
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1

        ## enum: MPCformer, our, vanillaGPT, thex
        self.accelarate_type=method
        self.gen_type=gen_type # enum: vanilla, embedReSend
        # self.gen_type="embedReSend" # enum: vanilla, embedReSend
        self.prefix_length=prefix_len
        self.gen_len=self.sequence_length-1
        self.device=device
    def __display__(self):
        t=""
        for x,y in self.__dict__.items():
            t+=f"{x}: {y} \t{type(y)}\n"
        return t

    def __str__(self):
        return self.__display__()

# # Inference arguments
# class config():
#    def __init__(self):
#        self.batch_size = 1
#        self.num_hidden_layers = 12
#        self.hidden_size = 768
#        self.intermediate_size = self.hidden_size * 4
       
#        self.sequence_length=512
#        self.max_position_embeddings = self.sequence_length
#        #self.hidden_act = "newGeLU"
#        #self.softmax_act = "softmax"
#        self.hidden_act = "quad"
#        self.softmax_act = "softmax_2QUAD"
#        self.layer_norm_eps = 1e-12
#        self.num_attention_heads = 12
#        self.vocab_size = 50257
#        self.hidden_dropout_prob = 0.1
#        self.attention_probs_dropout_prob = 0.1

#        ## enum: MPCformer, our
#        self.accelarate_type="MPCformer"
#        self.accelarate_type="our"
#        self.gen_type="vanilla" # enum: vanilla, embedReSend
#        # self.gen_type="embedReSend" # enum: vanilla, embedReSend
#        self.prefix_length=16
#        self.gen_len=self.sequence_length-self.prefix_length
#        self.device=device

config = config()
print(f"using model config: {config}")


crypten.init()
cfg.communicator.verbose = True

# setup fake data for timing purpose
commInit = crypten.communicator.get().get_communication_stats()
input_ids = F.one_hot(torch.randint(low=0, high=config.vocab_size,
            size=(config.batch_size, config.prefix_length)), config.vocab_size).float().to(device)

print("init done")
timing = defaultdict(float)

if config.accelarate_type=="our" or config.accelarate_type=="onlyMM":
    m = GPTBaseFlatten(config, timing)
    model = encrypt_model(m, GPTBaseFlatten,
                  (config, timing), input_ids).eval()
else:
    m = gpt(config, timing)
    model = encrypt_model(m, gpt,
                  (config, timing), input_ids).eval()

# encrpy inputs
dec_ids = encrypt_tensor(input_ids[:,:2],config)
input_ids = encrypt_tensor(input_ids,config)

num=5
avg_t = defaultdict(float)
res_ls=[]

if config.accelarate_type=="our" or config.accelarate_type=="onlyMM":
    if config.gen_type=="vanilla":
        for i in tqdm(range(num)):
            m.reset_timing()
            time_s = time.time()
            # run a forward pass
            with crypten.no_grad():
                model.generate_vanilla(input_ids,dec_ids)

            time_e = time.time()
            timing["total_time"] = (time_e - time_s)
            res_ls.append(deepcopy(timing))
            for k,v in timing.items():
                avg_t[k]+=v
            print(timing)
    else:
        for i in tqdm(range(num)):
            m.reset_timing()
            time_s = time.time()
            # run a forward pass
            with crypten.no_grad():
                model.generate_ourmethod(input_ids,dec_ids)

            time_e = time.time()
            timing["total_time"] = (time_e - time_s)
            res_ls.append(deepcopy(timing))
            for k,v in timing.items():
                avg_t[k]+=v
            print(timing)
else:
    if config.gen_type=="vanilla":
        for i in tqdm(range(num)):
            m.reset_timing()
            time_s = time.time()
            # run a forward pass
            with crypten.no_grad():
                model.generate(input_ids,dec_ids, config.gen_len)
            time_e = time.time()
            timing["total_time"] = (time_e - time_s)
            res_ls.append(deepcopy(timing))
            for k,v in timing.items():
                avg_t[k]+=v
            print(timing)
    else:
        for i in tqdm(range(num)):
            m.reset_timing()
            time_s = time.time()
            # run a forward pass
            with crypten.no_grad():
                model.generate_ourmethod(input_ids,dec_ids, config.gen_len)
            time_e = time.time()
            timing["total_time"] = (time_e - time_s)
            res_ls.append(deepcopy(timing))
            for k,v in timing.items():
                avg_t[k]+=v
            print(timing)
def mean(ls):
   return sum(ls)/len(ls)

def var(ls):
   return np.var(ls)

def getMeanStdMinMax(res_ls):
   ls_of_each_att={}
   for k in res_ls[0].keys():
      ls_of_each_att[k]=[]

   for k in ls_of_each_att.keys():
      ls_of_each_att[k]=[x[k] for x in res_ls]
   # thus each key (e.g. linear time) have a list of values

   ## statistic the meanvalue,the std, the min, and the max
   keyls_t1=["EmbedTime","LinearTime","SoftmaxTime","ActivTime",
             "GenerationTime","total_time",]
   keyls_ct1=["EmbedCommTime","LinearCommTime","SoftmaxCommTime",
              "ActivCommTime","GenerationCommTime","total_time",]
   keyls_cb=["ËmbedCommByte","LinearCommByte","SoftmaxCommByte",
              "ActivCommByte","GenerationCommByte"]
   metric_cal(ls_of_each_att,keyls_t1)
   metric_cal(ls_of_each_att,keyls_ct1)
   metric_cal(ls_of_each_att,keyls_cb)

def metric_cal(ls_of_each_att,keyls_t1):
   tmp_mean_res=[]
   tmp_std_res=[]
   tmp_min_res=[]
   tmp_max_res=[]
   print(f"keys:{keyls_t1}")
   for k in keyls_t1:
      if k not in ls_of_each_att.keys():
         print(f"NotFound: {k}")
         tmp_mean_res.append(-1)
         tmp_std_res.append(-1)
         tmp_min_res.append(-1)
         tmp_max_res.append(-1)
      else:
         tmp_mean_res.append(mean(ls_of_each_att[k]))
         tmp_std_res.append(var(ls_of_each_att[k]))
         tmp_min_res.append(min(ls_of_each_att[k]))
         tmp_max_res.append(max(ls_of_each_att[k]))

   tmp_mean_res=[str(x) for x in tmp_mean_res]
   tmp_std_res=[str(x) for x in tmp_std_res]
   tmp_min_res=[str(x) for x in tmp_min_res]
   tmp_max_res=[str(x) for x in tmp_max_res]

   print("=====MEAN====")
   print("\t".join(tmp_mean_res))
   print("=====VAR====")
   print("\t".join(tmp_std_res))
   print("=====MIN====")
   print("\t".join(tmp_min_res))
   print("=====MAX====")
   print("\t".join(tmp_max_res))

getMeanStdMinMax(res_ls)

for k,v in avg_t.items():
   avg_t[k]/=num
   if "Byte" in k:
      avg_t[k]/=1024
      avg_t[k]/=1024
print("-------------")
print(avg_t)




## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")

