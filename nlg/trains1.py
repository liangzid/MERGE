"""
======================================================================
TRAINS1 ---

Stage 1 training, which use the vanilla language model cross entropy
training fucntion.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created:  9 二月 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp
from tqdm import tqdm

from datasets import load_dataset
from torch.utils.data import DataLoader

from transformersV4251.models.gpt2.gpt2_new import \
    GPT2LMHeadModel as BFSCNew
from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

import json
import numpy as np
import argparse

def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int,
                        required=False)
    parser.add_argument('--lr', default=3e-4, type=float,
                        required=False)
    parser.add_argument('--cuda_num', default='6', type=str, required=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        required=False)
    parser.add_argument('--task', default="cola", type=str,
                        required=True,)
    parser.add_argument("--max_seq_length", default=128,
                        type=int, required=False)
    parser.add_argument("--train", default=1, type=int,
                        required=True,)
    parser.add_argument('--pretrained_model_path', default='bert-tiny',
                        type=str, required=True,)
    parser.add_argument('--root_dir', default='/home/liangzi/mpcGen/nlg/',
                        type=str, required=False,)

    return parser.parse_args()

def get_pretrained_dataset(tokenizer,
                           max_sentence_length=1024,
                           task="wikitext",
                           subset="wikitext-103-v1"):
    """
    Only for Pretrain NLG corpus, consisting of:
        subset: wikitext-103-v1, wikitext-2-v1
    """

    def getSet(split="train"):
        train_set=load_dataset(task,subset,split=split)
        train_t=[x["text"] for x in train_set]
        res=tokenizer(train_t,padding="max_length",
                      truncation=True,
                    max_length=1024,return_tensors="pt")
        dset=TensorDataset(res.input_ids,
                        )
        return dset
    return getSet("train"),getSet("validation"),getSet("test")

def getFinetunedSet(tokenizer,
                    max_sentence_length=256,
                    task="GEM/web_nlg",subset="en"):
    """
    For Downstream Tasks based on Conditional Generation.
    task and subtask enums:
    + GEM/web_nlg
        + en
        + ru
    + e2e_nlg, subset:none
    """
    sep_token="<|sep|>"
    # sep_token=tokenizer.sep_token
    eos_token=tokenizer.eos_token

    def getSet(split="train"):
        train_set=load_dataset(task,subset,split=split)
        inps=[x["input"] for x in train_set]
        inps=[";".join(x) for x in inps]
        outs=[x["target"] for x in train_set]
        outs=[inps[i]+sep_token+outs[i]+eos_token\
              for i in range(len(train_set))]

        outss=tokenizer(outs,padding="max_length",
                      truncation=True,
                    max_length=max_sentence_length,return_tensors="pt")

        dset=TensorDataset(outss.input_ids,
                           )
        return dset
    return getSet("train"),getSet("validation"),getSet("test")
    
def trainConditional(model,
          optimizer,
          train_loader,
          val_loader,
          task,
          save_path,
          EPOCH,LR,DEVICE,
          batch_size=32,
          ):

    ii=0
    past_losses=10000
    tqdm1=tqdm(total=EPOCH)
    for epoch in range(EPOCH):
        tqdm1.update(1)

        print(f"-------EPOCH {epoch}-------------")
        for i,(inps,) in enumerate(train_loader):
            print(ii)
            ii+=1
            inps,=inps.to(DEVICE),

            outputs = model(inps,
                            labels=inps)

            loss = outputs.loss
            logits=outputs.logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ii%300==0:
                print(f"loss:{loss.item()}")

            if ii%10000==0:
                print("Run Validating...")
                losses=test(test_loader=val_loader,
                         model=model,
                         task=task,
                         batch_size=batch_size,
                         DEVICE=DEVICE)
                print(f">>Val Loss: {losses}")

                if losses<past_losses:
                    model.save_pretrained(save_path)
                    past_losses=losses

    print("End Training.")
            
def test(test_loader,model,task,batch_size=32,DEVICE="cpu"):
    model.eval()
    losses=0.

    with torch.no_grad():
        for i,(inps,) in enumerate(test_loader):
            inps,=inps.to(DEVICE),

            outputs = model(inps,labels=inps)
            loss = outputs.loss
            losses+=loss

    losses/=((i+1)*batch_size)
        
    model.train()
    return losses

## TODO: MOVE these functions to =inference.py=
# def infer_vanilla(test_loader,model,task,DEVICE="cpu"):
#     model.eval()

#     evaluate_dict={"GEM/web_nlg":["bleu","meteor","chrf","TER",
#                                   "bertscore","bleurt"],
#                    "ele_nlg":["bleu","NIST","METEOR","ROUGE_L","CIDEr"]}

# def infer_reSend(test_loader,model,task,DEVICE="cpu"):
#     pass

def main():
    EPOCH = 5
    # LR = 5e-5 
    LR = 5e-5 
    DEVICE = torch.device("cuda:5")
    # DEVICE = torch.device("cpu")
    BATCH_SIZE =1
    batch_size=BATCH_SIZE
    task_ls=["GEM/web_nlg","e2e_nlg"]
    subtaskls=["en",None]

    task="GEM/web_nlg"
    subtask="en"

    PATH = f'./stage1_ckpts/{task}-epoch{EPOCH}-lr{LR}-bs{BATCH_SIZE}'

    prefix_path="/home/liangzi/models/"
    model_name="gpt2/"
    frmpth=prefix_path+model_name
    
    model = BFSCNew.from_pretrained(frmpth)
    # model = AutoModelForCausalLM.from_pretrained(frmpth)
    tokenizer = AutoTokenizer.from_pretrained(frmpth)
    tokenizer.pad_token="<|pad|>"
    tokenizer.sep_token="<|sep|>"

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model = model.to(DEVICE)

    trs,vas,tes=getFinetunedSet(tokenizer,128,task,subtask)
    print(f"train set len: {len(trs)}")
    print(f"validation set len: {len(vas)}")
    print(f"test set len: {len(tes)}")

    print(f"batch_size: {batch_size}")
    trloader=DataLoader(trs,batch_size=batch_size,
                            shuffle=True,drop_last=False)
    valoader=DataLoader(vas,batch_size=batch_size,
                            shuffle=True,drop_last=True)
    teloader=DataLoader(tes,batch_size=batch_size,
                            shuffle=True,drop_last=True)

    #============================================
    trainConditional(model, optimizer,
                     trloader,valoader,
                     task,
                     PATH,
                     batch_size=BATCH_SIZE,
          EPOCH=EPOCH,LR=LR,
          DEVICE=DEVICE,)
    tokenizer.save_pretrained(PATH)
    #============================================

    model=model.from_pretrained(PATH)
    model.to(DEVICE)
    model.eval()
    
    # test(test_loader=val_loader,model=model,task=task,
    #      batch_size=BATCH_SIZE,DEVICE=DEVICE)

    test(test_loader=test_loader,model=model,task=task,
         batch_size=BATCH_SIZE,DEVICE=DEVICE)


## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


