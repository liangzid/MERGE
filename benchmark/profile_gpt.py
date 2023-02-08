import sys
import os
import time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import AutoConfig, BertForSequenceClassificationWrapper

import crypten
import crypten.communicator as comm
from crypten.config import cfg
from utils import encrypt_tensor, encrypt_model

from gpt import gpt
from encryp_decoder import GPTBaseFlatten

# Inference arguments
class config():
   def __init__(self):
       self.batch_size = 1
       self.num_hidden_layers = 12
       self.hidden_size = 768
       self.intermediate_size = self.hidden_size * 4
       
       self.sequence_length=512
       self.max_position_embeddings = self.sequence_length
       #self.hidden_act = "newGeLU"
       #self.softmax_act = "softmax"
       self.hidden_act = "quad"
       self.softmax_act = "softmax_2QUAD"
       self.layer_norm_eps = 1e-12
       self.num_attention_heads = 12
       self.vocab_size = 50257
       self.hidden_dropout_prob = 0.1
       self.attention_probs_dropout_prob = 0.1

       ## enum: MPCformer, our
       self.accelarate_type="MPCformer"
       self.accelarate_type="our"
       self.gen_type="vanilla" # enum: vanilla, embedReSend
       # self.gen_type="embedReSend" # enum: vanilla, embedReSend
       self.prefix_length=16
       self.gen_len=self.sequence_length-self.prefix_length

config = config()
print(f"using model config: {config}")

# 2PC setting
rank = sys.argv[1]
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(2)
os.environ["MASTER_ADDR"] = "219.245.186.45"
os.environ["MASTER_PORT"] = "29500"
os.environ["RENDEZVOUS"] = "env://"

crypten.init()
cfg.communicator.verbose = True

# setup fake data for timing purpose
commInit = crypten.communicator.get().get_communication_stats()
input_ids = F.one_hot(torch.randint(low=0, high=config.vocab_size,
            size=(config.batch_size, config.prefix_length)), config.vocab_size).float().cuda()

timing = defaultdict(float)

if config.accelarate_type=="MPCformer":
    m = gpt(config, timing)
    model = encrypt_model(m, gpt,
                  (config, timing), input_ids).eval()
else:
    m = GPTBaseFlatten(config, timing)
    model = encrypt_model(m, GPTBaseFlatten,
                  (config, timing), input_ids).eval()

# encrpy inputs
input_ids = encrypt_tensor(input_ids)

num=10
avg_t = defaultdict(float)

if config.accelarate_type=="MPCformer":
    for i in tqdm(range(10)):
        m.reset_timing()
        time_s = time.time()
        # run a forward pass
        with crypten.no_grad():
            model.generate(input_ids, config.gen_len)
        time_e = time.time()
        timing["total_time"] = (time_e - time_s)
        for k,v in timing.items():
            avg_t[k]+=v
        print(timing)
else:
    if config.gen_type=="vanilla":
        for i in tqdm(range(10)):
            m.reset_timing()
            time_s = time.time()
            # run a forward pass
            with crypten.no_grad():
                model.generate_vanilla(input_ids)

            time_e = time.time()
            timing["total_time"] = (time_e - time_s)
            for k,v in timing.items():
                avg_t[k]+=v
            print(timing)
    else:
        for i in tqdm(range(10)):
            m.reset_timing()
            time_s = time.time()
            # run a forward pass
            with crypten.no_grad():
                model.generate_ourmethod(input_ids)

            time_e = time.time()
            timing["total_time"] = (time_e - time_s)
            for k,v in timing.items():
                avg_t[k]+=v
            print(timing)

for k,v in avg_t.items():
   avg_t[k]/=num
print("-------------")
print(avg_t)
