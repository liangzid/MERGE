"""
======================================================================
GETTESTSETPREFIXLENGENLEN ---

This file was used to obtain the prefix lengths and the text lengths
of several corpus.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 17 四月 2023
======================================================================
"""

# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

from numpy.core.numeric import outer
from tqdm import tqdm

from datasets import load_dataset
from torch.utils.data import DataLoader

from transformersV4251.models.gpt2.gpt2_new import \
    GPT2LMHeadModel as BFSCNew
from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM
from transformers import T5ForConditionalGeneration
from transformers import BartForConditionalGeneration
from transformers import AutoTokenizer,AutoConfig

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch import tensor

import json
import numpy as np
import argparse

from trains1 import *



def getPrefixLen(dataset="multiwoz_nlg",
                 model_path="/home/liangzi/models/gpt2/",
                 save_path="blablabla.json"):
    subset=None
    task=dataset
    msl=128
    withsep=True
    padding="max_length"

    if "t5" or "bart" in model_path:
        withsep=False

    tokenizer=AutoTokenizer.from_pretrained(model_path)

    if task=="common_gen":
        te=getTestDataSet(tokenizer,split="validation",
                                max_sentence_length=msl,
                                task=task,subset=subset,withsep=withsep,
                          )
    else:
        te=getTestDataSet(tokenizer,split="test",
                                max_sentence_length=msl,
                                task=task,subset=subset,withsep=withsep,
                          )

    prefix,labels=te
    lens_p=[len(p[0]) for p in prefix]
    lens_l=[len(l[0]) for l in prefix]

    with open(save_path, 'w',encoding='utf8') as f:
        json.dump({"prefix":lens_p,
                   "labels":lens_l},f,
                  ensure_ascii=False,indent=4)
    print(f"save to {save_path} done.")


def main():
    model_path="./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/"
    for x in ["multiwoz_nlg","common_gen","daily_dialog",]:
        
        getPrefixLen(dataset=x,
                     model_path=model_path,
                     save_path=f"../{x}_test_lens.json")

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


