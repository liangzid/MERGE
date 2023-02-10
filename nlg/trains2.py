"""
======================================================================
TRAINS2 ---

After train teacher model with `trains1.py`, we should:
1. Calculate the Constant Matrix, and load the modified GPT-2 model.
2. Distill the modified version GPT-2 with the teacher model and the
train set.

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

from datasets import load_dataset
from torch.utils.data import DataLoader

# todo: need to changed it.
from transformersV4251.models.bert.modeling_bert_new import \
    BertForSequenceClassification as BFSCNew

from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

import json
import numpy as np
import argparse

from trains1 import getFinetunedSet,test

def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int,
                        required=False,)
    parser.add_argument('--lr', default=3e-4, type=float,
                        required=False,)
    parser.add_argument('--cuda_num', default='6', type=str, required=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        required=False,)
    parser.add_argument('--task', default="cola", type=str,
                        required=True,)
    parser.add_argument("--max_seq_length", default=128,
                        type=int, required=False,)
    parser.add_argument("--train", default=1, type=int,
                        required=True,)

    parser.add_argument('--teach_ckpt', default='bert-tiny',
                        type=str, required=True,)
    parser.add_argument('--stu_ckpt', default='bert-tiny',
                        type=str, required=True,)
    parser.add_argument('--stu_save_ckpt', default='bert-tiny',
                        type=str, required=True,)

    parser.add_argument('--using_entropy', default=1,
                        type=int, required=False,)
    parser.add_argument('--using_softLabel', default=0,
                        type=int, required=False,)
    parser.add_argument('--tau', default=1.,
                        type=float, required=False,)
    parser.add_argument('--using_interKL', default=0,
                        type=int, required=False,)
    parser.add_argument('--using_wordEmbedMSE', default=0,
                        type=int, required=False,)
    parser.add_argument('--root_dir', default='/home/liangzi/he_transformer/newglue/',
                        type=str, required=False,)
    return parser.parse_args()


def train(args, tmodel, smodel,
          optimizer, train_loader, val_loader,
          task,
          EPOCH,LR,DEVICE,
          batch_size=32,
          ):
    kl_loss=torch.nn.KLDivLoss(reduction='batchmean')

    ii=0
    for epoch in range(EPOCH):
        ii+=1

        print(f"-------EPOCH {epoch}-------------")
        for i,(inps,atts) in enumerate(train_loader):
            inps,atts=inps.to(DEVICE),\
                atts.to(DEVICE)

            toutputs=tmodel(inps,attention_mask=atts,labels=inps,
                            output_hidden_states=True)
            teacher_logits=toutputs.logits

            outputs = smodel(inps,attention_mask=atts,
                             labels=inps,
                             output_hidden_states=True)

            entropy_loss=0.
            softlabel_loss=0.
            inter_loss=0.
            wordEmMSE_loss=0.
            if args.using_entropy==1:
                entropy_loss = outputs.loss
            if args.using_softLabel==1:
                # print("Original logits",teacher_logits[0])

                # new_stu_logits=F.log_softmax(outputs.logits/args.tau,dim=1)
                # new_tea_logits=F.log_softmax(teacher_logits/args.tau,dim=1)

                # print("tau",type(args.tau),args.tau)

                new_stu_logits=F.softmax(outputs.logits/args.tau,dim=1)
                new_tea_logits=F.softmax(teacher_logits/args.tau,dim=1)
                # print("newloss",new_stu_logits[0])

                # print(new_stu_logits.shape,new_tea_logits.shape)
                softlabel_loss=kl_loss(new_stu_logits.log(),
                                new_tea_logits,
                                )
                # print("softlabelLoss",softlabel_loss)
                softlabel_loss*=args.tau**2

            if args.using_interKL==1:
                ## we use MSE loss for that.
                mse_loss=0.
                lens=len(toutputs.hidden_states)
                for j in range(lens):
                    mse_loss+=F.mse_loss(toutputs.hidden_states[j],
                               outputs.hidden_states[j],reduction="mean")
                mse_loss/=lens

            # todo
            if args.using_wordEmbedMSE==1:
                wordEmMSE_loss=0.
            
            a1,a2,a3=0.25,0.25,0.25
            a4=1-a1-a2-a3
            loss = a1*entropy_loss + a2*softlabel_loss +\
                a3*inter_loss + a4*wordEmMSE_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%300==0:
                # print(f"loss:{loss.item()}")
                print(f"Loss:{loss}\tEntropy:{entropy_loss}\
                \tDistill:{softlabel_loss}\tInter:{inter_loss}\
                \twordEmbed{wordEmMSE_loss}")

            if ii%500==0:
                print("Run Validating...")
                losses=test(test_loader=val_loader,
                         model=smodel,
                         task=task,
                         batch_size=batch_size,
                         DEVICE=DEVICE)

                if losses<past_losses:
                    smodel.save_pretrained(args.stu_save_ckpt)
                    past_losses=losses

    print("End Training.")

def main1():
    args=setup_train_args()
    
    EPOCH = args.epochs
    LR = args.lr
    if args.cuda_num=="cpu":
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(f"cuda:{args.cuda_num}")
    BATCH_SIZE =args.batch_size
    task=args.task
    
    tmodel = AutoModelForCausalLM.from_pretrained(args.teach_ckpt)
    ttokenizer = AutoTokenizer.from_pretrained(args.teach_ckpt)

    ## TODO: change the name of self-defined class.
    smodel = BFSCNew.from_pretrained(args.stu_ckpt)
    stokenizer = AutoTokenizer.from_pretrained(args.stu_ckpt)
    tokenizer=ttokenizer

    optimizer = torch.optim.AdamW(smodel.parameters(), lr=LR)
    smodel = smodel.to(DEVICE)
    tmodel = tmodel.to(DEVICE)

    trs,vas,tes=getFinetunedSet(tokenizer,256,task,subtask)

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
              DEVICE=DEVICE,)
        # model.save_pretrained(PATH)
        tokenizer.save_pretrained(args.stu_save_ckpt)
        #============================================

    smodel=smodel.from_pretrained(args.stu_save_ckpt)
    smodel.to(DEVICE)
    smodel.eval()
    
    # test(test_loader=val_loader,model=smodel,task=task,
    #      batch_size=BATCH_SIZE,DEVICE=DEVICE)

    print("Now on Original Student Model.")
    smodel=smodel.from_pretrained(args.stu_ckpt)
    smodel.to(DEVICE)
    smodel.eval()
    
    test(test_loader=teloader,model=smodel,task=task,
         batch_size=BATCH_SIZE,DEVICE=DEVICE)


## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


