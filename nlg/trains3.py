"""
======================================================================
TRAINS3 ---

Stage 3 of training. Stage 2 of knowledge distillation.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 14 二月 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

from transformersV4251.models.gpt2.gpt2_new import \
    GPT2LMHeadModel as BFSCNew
from transformersV4251.models.t5.modeling_t5 import \
    T5ForConditionalGeneration as T5New
from transformersV4251.models.bart.modeling_bart import \
    BartForConditionalGeneration as BartNew

from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM
from transformers import T5ForConditionalGeneration
from transformers import BartForConditionalGeneration
from transformers import AutoTokenizer
from transformers import AutoConfig
from torch.utils.data import DataLoader, TensorDataset

from datasets import load_dataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import tensor

import json
import numpy as np
import argparse

from trains1 import getFinetunedSet,test
from trains2 import setup_train_args, train 

def main():
    args=setup_train_args()
    torch.autograd.set_detect_anomaly(True)
    
    EPOCH = args.epochs
    LR = args.lr
    if args.cuda_num=="cpu":
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(f"cuda:{args.cuda_num}")
    BATCH_SIZE =args.batch_size
    batch_size=args.batch_size
    task=args.task
    if task=="web_nlg":
        subtask="release_v2"
    elif task=="e2d_nlg":
        subtask=None
    else:
        subtask=None

    frmpth=args.teach_ckpt
    if "gpt" in frmpth:
        only_decoder=True
    elif "t5" in frmpth or "bart" in frmpth:
        only_decoder=False
    else:
        only_decoder=True
    print(f"The Backbone is Only a Decoder: {only_decoder}.")
    
    if "t5" in args.teach_ckpt:
        tmodel = T5New.\
            from_pretrained(args.teach_ckpt)
    elif "bart" in args.teach_ckpt:
        tmodel = BartNew.\
            from_pretrained(args.teach_ckpt)
    else:
        tmodel = BFSCNew.from_pretrained(args.teach_ckpt)
    print("TEA Original embedding size: ",tmodel.get_input_embeddings().weight.shape[0])
    ttokenizer = AutoTokenizer.from_pretrained(args.teach_ckpt)
    tmodel.resize_token_embeddings(len(ttokenizer))

    config=AutoConfig.from_pretrained(args.stu_ckpt)
    if args.using_quadacti==1:
        config.activation_function="quad" # set to quad activation
    else:
        config.activation_function="gelu_new" # set to quad activation
    if args.using_simLN==1:
        config.layerNormType="sim" # set to quad activation
    else:
        config.layerNormType="no-sim" # set to quad activation
    config.save_pretrained(args.stu_ckpt)
    config.save_pretrained(args.stu_save_ckpt)
    
    if "t5" in args.teach_ckpt:
        smodel = T5New.\
            from_pretrained(args.stu_ckpt)
    elif "bart" in args.teach_ckpt:
        smodel = BartNew.\
            from_pretrained(args.stu_ckpt)
    else:
        smodel = BFSCNew.from_pretrained(args.stu_ckpt)
    print("STU Original embedding size: ",smodel.get_input_embeddings().weight.shape[0])

    # print(smodel.transformer.h[2].attn.M)

    # ## re-initialize the LayerNorm parameters.
    # if args.using_simLN==1:
    #     for k,param in smodel.named_parameters():
    #         if "ln_" in k or "layer_norm" in k\
    #         or "layernorm" in k: # which means the layerNorm
    #             if "weight" in k:
    #                 param=torch.ones_like(param)*1e-3
    #             if "bias" in k:
    #                 param=torch.zeros_like(param)

    stokenizer = AutoTokenizer.from_pretrained(args.stu_ckpt)
    tokenizer=ttokenizer
    smodel.resize_token_embeddings(len(tokenizer))
    print("length of vocab in tokenizer: ",len(tokenizer))

    if args.using_wordEmbedMSE==1:
        ## using the embedding representation as the classifier params.
        embedding_weight=smodel.get_input_embeddings().weight.T
        d,v=embedding_weight.shape
        print(f"V: {v}\td: {d}")
        newlm=nn.Linear(d,v,bias=False)
        newlm.weight=nn.Parameter(embedding_weight.T)
        for param in newlm.parameters():
            param.requires_grad = False
        smodel.set_output_embeddings(newlm)
        print("whether the last linear map has grad: ",
              smodel.lm_head.weight.requires_grad)
        for name,param in smodel.named_parameters():
            if "wte" in name:
                print("find word embedding layer,\
                now set the grad to false.")
                param.required_grad=False
        print("whether the embedding layer has grad: ",
              False)

    optimizer = torch.optim.AdamW(smodel.parameters(), lr=LR,
                                  weight_decay=args.weight_decay,)
    smodel = smodel.to(DEVICE)
    tmodel = tmodel.to(DEVICE)

    print(f"max sequence length: {args.max_seq_length}")
    trs,vas,tes=getFinetunedSet(tokenizer,args.max_seq_length,
                                task,subtask,only_decoder)

    trloader=DataLoader(trs,batch_size=batch_size,
                            shuffle=True,drop_last=False)
    valoader=DataLoader(vas,batch_size=batch_size,
                            shuffle=True,drop_last=True)
    teloader=DataLoader(tes,batch_size=batch_size,
                            shuffle=True,drop_last=True)

    if args.train==1:
        #============================================
        train(args,tmodel=tmodel,smodel=smodel, optimizer=optimizer,
              train_loader=trloader,val_loader=valoader,
              task=task,
              batch_size=BATCH_SIZE,
              EPOCH=EPOCH,LR=LR,
              DEVICE=DEVICE,
              only_decoder=only_decoder)
        # model.save_pretrained(PATH)
        tokenizer.save_pretrained(args.stu_save_ckpt)
        #============================================

    smodel=smodel.from_pretrained(args.stu_save_ckpt)
    smodel.to(DEVICE)
    smodel.eval()
    
    # test(test_loader=val_loader,model=smodel,task=task,
    #      batch_size=BATCH_SIZE,DEVICE=DEVICE)

    print("Now on Original Student Model.")
    smodel=smodel.from_pretrained(args.stu_save_ckpt)
    smodel.to(DEVICE)
    smodel.eval()
    
    test(test_loader=teloader,model=smodel,task=task,
         batch_size=BATCH_SIZE,DEVICE=DEVICE)

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")
