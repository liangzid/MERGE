"""
======================================================================
CALCULATE_PARAMS --- 

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 28 四月 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

## transformers related import
from transformers import T5Tokenizer,T5ForConditionalGeneration
from transformers import BertTokenizer
from transformers import pipeline
import transformers


def main():
    model_path="/home/liangzi/models/t5-small"
    config=transformers.AutoConfig.from_pretrained(model_path)

    layers=[6,12,12,24,48,48]
    d_ls=[512,768,1024,1024,2048,4096]
    head_ls=[8,12,16,16,32,64]

    i=6
    config.d_model,config.num_heads,config.num_layers=\
        d_ls[i],head_ls[i],layers[i]

    config.d_kv=768//12
    config.d_ff=768*4
    config.vocab_size=50257

    model=transformers.T5ForConditionalGeneration(config)
    # model=transformers.T5ForConditionalGeneration.from_config(config)
    
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of all parameters: {}'\
            .format(num_parameters/1e6))

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


