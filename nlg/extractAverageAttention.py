"""
======================================================================
EXTRACTAVERAGEATTENTION ---

Extract and calculate the constant attention matrix.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 10 二月 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp
from copy import deepcopy,copy

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer,AutoConfig
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset
import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm

from trains1 import getFinetunedSet
from transformersV4251.models.gpt2.gpt2_new import \
    GPT2LMHeadModel as BFSCNew

def calAttnMat(task,ckpt_path,ckpt_save_p):
    ## 0. config varible
    config=AutoConfig.from_pretrained(ckpt_path)
    batch_size=1
    msl=128
    device="cpu"
    num_layer=config.num_hidden_layers
    num_head=config.num_attention_heads

    ## 1.2. load train set and checkpoint
    tokenizer=AutoTokenizer.from_pretrained(ckpt_path)
    model=AutoModelForCausalLM.from_pretrained(ckpt_path)
    model.resize_token_embeddings(len(tokenizer))
    # model=BFSCNew.from_pretrained(ckpt_path)
    model.eval()
    model.to(device)

    task,subset="GEM/web_nlg","en"
    dataset=getFinetunedSet(tokenizer,
                            msl,
                            task=task,subset=subset,)[0]
    # dataset=Subset(dataset, np.arange(5000))
    
    ## 3. running forward inference and record the attention matrics
    attentions=[torch.zeros((num_head,msl,msl),
            dtype=torch.float64) for _ in range(num_layer)]
    final_atts=[torch.zeros((num_head,msl,msl),
            dtype=torch.float64) for _ in range(num_layer)]
    
    nums=len(dataset)
    print("length of dataset: ",nums)
    block=2500
    num_block=nums//block
    num_block=1

    for nb in tqdm(range(num_block)):
        littleset=Subset(dataset, np.arange(block))
        print(">>>little set length: ",len(littleset))
        batch_size=1
        train_loader=DataLoader(littleset,batch_size=batch_size,
                                shuffle=True,drop_last=False)
        for i, (inps,) in enumerate(train_loader):
            inps,=inps.to(device),
            atts=model(inps,labels=inps,
                        output_attentions=True).attentions

            for j,a in enumerate(atts):
                # print("attention shape:>>>",a.shape) # 1,12,128,128
                # print(j)
                attentions[j]+=a[0]
            del atts

        for iii in range(num_layer):
            attentions[iii]/=block
            final_atts[iii]+=attentions[iii]
        attentions=[torch.zeros((num_head,msl,msl),
            dtype=torch.float64) for _ in range(num_layer)]

        del train_loader
        if len(dataset)<block:
            break
        else:
            dataset=dataset[block:]
    
    ## 4. set new model parameter and save it. 
    newmodel=model
    names=[]
    for name,params in model.named_parameters():
        names.append(name)
    # print(names)
    old_state_dict=newmodel.state_dict()
    old_state_dict=dict(old_state_dict)
    print(old_state_dict.keys())
    for kk in range(num_layer):
        kyname=f"transformer.h.{kk}.attn.M"
        old_state_dict[kyname]=attentions[kk]
    print(old_state_dict.keys())

    newmodel=BFSCNew.from_pretrained(ckpt_path)
    newmodel.resize_token_embeddings(len(tokenizer))
    print("Now load constant Attention.")
    newmodel.load_state_dict(old_state_dict)
    print("Load constant Attention Done.")
    newmodel.save_pretrained(ckpt_save_p)
    tokenizer.save_pretrained(ckpt_save_p)
    print(f"Save DONE. Save to {ckpt_save_p}")

    # del train_loader
    del newmodel
    del old_state_dict
    del model

    # for i,(inps,atts,segs,labels) in enumerate(train_loader):
    #     if i>3:
    #         break
    #     ii+=1
    #     progress.update(1)
    #     inps,atts,segs,labels=inps.to(device),atts.to(device),\
    #         segs.to(device),labels.to(device)
    #     # atts=newmodel(inps,atts,segs,labels=labels,
    #     #               output_attentions=True).attentions

def main():
    task="GEM/web_nlg"
    # ckpt_path="./save_models/saved_bert-tiny_taskcola-epoch30-lr3e-05-bs32"
    # ckpt_path="./stage1_ckpts/GEM/web_nlg-epoch5-lr5e-05-bs1"
    ckpt_path="./stage1_ckpts/GEM/web_nlg-epoch3-lr5e-05-bs1"
    ckpt_save_p=ckpt_path+"___withConstantMatrix"
    calAttnMat(task,ckpt_path,ckpt_save_p)
        

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


