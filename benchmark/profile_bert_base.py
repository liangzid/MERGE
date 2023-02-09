"""
======================================================================
PROFILE_BERT_BASE ---

encryption time test for bert-base

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

import sys
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import AutoConfig, BertForSequenceClassificationWrapper

import crypten
import crypten.communicator as comm
from crypten.config import cfg
from utils import encrypt_tensor, encrypt_model

from models import Bert
from encryp_bert_base import BertBaseFlatten
from models import BertEmbeddings

# Inference arguments
class config():
   def __init__(self):
       self.batch_size = 1
       self.num_hidden_layers = 12
       self.hidden_size = 768
       self.intermediate_size = 3072
       self.sequence_length = 512
       self.max_position_embeddings = 512
       self.hidden_act = "quad"
       self.softmax_act = "softmax_2RELU"
       self.layer_norm_eps = 1e-12
       self.num_attention_heads = 12
       self.vocab_size = 28996
       self.hidden_dropout_prob = 0.1
       self.attention_probs_dropout_prob = 0.1
       self.device="cuda:0"

   def __display__(self):
      t=""
      for x,y in self.__dict__.items():
         t+=f"{x}: {y} \t{type(y)}\n"
      return t
         
   def __str__(self):
      return self.__display__()

config = config()
print(f"using model config: {config}")

# 2PC setting
rank = sys.argv[1]
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(2)
os.environ["MASTER_ADDR"] = "219.245.186.45"
os.environ["MASTER_PORT"] = "10001"
os.environ["RENDEZVOUS"] = "env://"

# eval_type="VanillaBert"
eval_type="FastBert"
# eval_type="testlinearlayer"

crypten.init()
# show details of the communication procedure
cfg.communicator.verbose = True

# setup fake data for timing purpose
commInit = crypten.communicator.get().get_communication_stats()
input_ids = F.one_hot(torch.randint(low=0, high=config.vocab_size, size=(config.batch_size, config.sequence_length)), config.vocab_size).float().to(config.device)

# input_ids = torch.zeros((config.batch_size,
#                          config.sequence_length,
#                          config.hidden_size)).float().to("cuda:0")

timing = defaultdict(float)
avg_t = defaultdict(float)

if eval_type=="VanillaBert":
    m = Bert(config, timing)
    model = encrypt_model(m, Bert, (config, timing), input_ids).eval()
elif eval_type == "FastBert":
    m = BertBaseFlatten(config, timing)
    model = encrypt_model(m, BertBaseFlatten,
                          (config, timing), input_ids).eval()
else:
    # test the effort of the linear layer.
    print("Open the Test Mode.")
    from encryp_only_for_test import BertTest
    m = BertTest(config, timing)
    model = encrypt_model(m, BertTest,
                          (config, timing), input_ids).eval()
    

model=model.to(config.device)

# encrpy inputs
input_ids = encrypt_tensor(input_ids,config)

num=10
for i in range(num):
    m.reset_timing()
    time_s = time.time()
    # run a forward pass
    with crypten.no_grad():
       res=model(input_ids)
       
    time_e = time.time()
    timing["total_time"] = (time_e - time_s)
    for k,v in timing.items():
       avg_t[k]+=v
    print(timing)

for k,v in avg_t.items():
   avg_t[k]/=num
   if "Byte" in k:
      avg_t[k]/=1024
      avg_t[k]/=1024
print("-------------")
print(avg_t)
