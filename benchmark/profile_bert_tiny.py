"""
Encryption version BERT tiny with precomuption.

Zi Liang, 0207
"""

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
from encryp_bert_tiny import BertTinyFlatten
from models import BertEmbeddings

# Inference arguments
class config():
   def __init__(self):
       self.batch_size = 1
       self.num_hidden_layers = 2
       self.hidden_size =128 
       self.intermediate_size =512 
       self.sequence_length = 512
       self.max_position_embeddings = 512
       self.hidden_act = "quad"
       self.softmax_act = "softmax_2RELU"
       self.layer_norm_eps = 1e-12
       self.num_attention_heads = 2
       self.vocab_size = 30522
       self.hidden_dropout_prob = 0.1
       self.attention_probs_dropout_prob = 0.1

config = config()
print(f"using model config: {config}")

# 2PC setting
rank = sys.argv[1]
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(2)
os.environ["MASTER_ADDR"] = "219.245.186.45"
os.environ["MASTER_PORT"] = "29500"
os.environ["RENDEZVOUS"] = "env://"

# eval_type="VanillaBert"
eval_type="FastBert"

crypten.init()
# show details of the communication procedure
cfg.communicator.verbose = True

# setup fake data for timing purpose
commInit = crypten.communicator.get().get_communication_stats()
input_ids = F.one_hot(torch.randint(low=0, high=config.vocab_size, size=(config.batch_size, config.sequence_length)), config.vocab_size).float().to("cuda:0")

# input_ids = torch.zeros((config.batch_size,
#                          config.sequence_length,
#                          config.hidden_size)).float().to("cuda:0")

timing = defaultdict(float)
avg_t = defaultdict(float)

if eval_type=="VanillaBert":
    m = Bert(config, timing)
    model = encrypt_model(m, Bert, (config, timing), input_ids).eval()
else:
    m = BertTinyFlatten(config, timing)
    model = encrypt_model(m, BertTinyFlatten,
                          (config, timing), input_ids).eval()

model=model.to("cuda:0")

# encrpy inputs
input_ids = encrypt_tensor(input_ids)

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
print("-------------")
print(avg_t)
