"""
======================================================================
EXTRACTAVERAGEATTENTION ---

Extract the attention matrix and average on the train set.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2022, ZiLiang, all rights reserved.
    Created: 25 十二月 2022
======================================================================
"""

# ------------------------ Code --------------------------------------

import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp
from copy import deepcopy,copy

from transformers import AutoModelForSequenceClassification 
from transformers import AutoTokenizer,AutoConfig
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn

from tqdm import tqdm

from train import getDataset
from transformersV4251.models.bert.modeling_bert_new import \
    BertForSequenceClassification as BFSCNew


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
    model=AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    # model=BFSCNew.from_pretrained(ckpt_path)
    model.eval()
    model.to(device)

    dataset=getDataset(tokenizer,{},msl=msl,glue_task=task,mode="train")
    
    ## 3. running forward inference and record the attention matrics
    train_loader=DataLoader(dataset,batch_size=batch_size,
                            shuffle=True,drop_last=False)

    attentions=[torch.zeros((num_head,msl,msl),
            dtype=torch.float64) for _ in range(num_layer)]

    ii=0
    progress=tqdm(total=len(dataset))
    for i,(inps,atts,segs,labels) in enumerate(train_loader):
        # if i>5:
            # break
        ii+=1
        progress.update(1)

        inps,atts,segs,labels=inps.to(device),atts.to(device),\
            segs.to(device),labels.to(device)
        atts=model(inps,atts,segs,labels=labels,
                      output_attentions=True).attentions

        for j,a in enumerate(atts):
            # print("attention shape:>>>",a.shape)
            # print(j)
            attentions[j]+=a[0]
    for iii in range(num_layer):
        attentions[iii]/=ii
    # print(attentions)
    
    ## 4. set new model parameter and save it. 
    newmodel=model
    names=[]
    for name,params in model.named_parameters():
        names.append(name)
    print(names)
    old_state_dict=newmodel.state_dict()
    old_state_dict=dict(old_state_dict)
    print(old_state_dict.keys())
    for kk in range(num_layer):
        kyname=f"bert.encoder.layer.{kk}.attention.self.M"
        old_state_dict[kyname]=attentions[kk]
    new_state_dict=copy(old_state_dict)
    for key in old_state_dict:
        if "key" in key or "query" in key:
            new_state_dict.pop(key)
    old_state_dict=new_state_dict
    print(old_state_dict.keys())


    newmodel=BFSCNew.from_pretrained(ckpt_path)
    print("Now load constant Attention.")
    newmodel.load_state_dict(old_state_dict)
    print("Load constant Attention Done.")
    newmodel.save_pretrained(ckpt_save_p)
    tokenizer.save_pretrained(ckpt_save_p)
    print(f"Save DONE. Save to {ckpt_save_p}")

    del train_loader
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
    task="cola"
    ckpt_path="./save_models/saved_bert-tiny_taskcola-epoch30-lr3e-05-bs32"
    ckpt_save_p=ckpt_path+"___withConstantMatrix"
    calAttnMat(task,ckpt_path,ckpt_save_p)

def main2():

    task_ls=["cola","mnli_matched","mnli_mismatched","mrpc","qnli",
             "qqp","rte","wnli"]
    models=["bert-tiny","bert-base-uncased"]

    for task in task_ls:
        for model in models:
            ckpt_path=f"./save_models/saved_{model}_task{task}-epoch30-lr3e-05-bs32"
            ckpt_save_p=ckpt_path+"___withConstantMatrix"
            print("=="*15)
            print(f"Task: {task}\t Model: {model}")
            print(f"From path: {ckpt_path}")
            print(f"Target path: {ckpt_save_p}")
            print("=="*15)
            calAttnMat(task,ckpt_path,ckpt_save_p)
        
## running entry
if __name__=="__main__":
    main()
    # main2()
    print("EVERYTHING DONE.")

