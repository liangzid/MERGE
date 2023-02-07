"""
======================================================================
DISTILL_TRAIN --- 

Distill student model from original teacher models.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2022, ZiLiang, all rights reserved.
    Created: 26 十二月 2022
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

from transformersV4251.models.bert.modeling_bert_new import \
    BertForSequenceClassification as BFSCNew

from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from transformers import AutoModelForSequenceClassification 
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score

from transformers import glue_compute_metrics,glue_output_modes,glue_tasks_num_labels
glue_output_modes["sst2"]="classification"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

import json
import numpy as np
import argparse

from train import preprocess,test

from transformersV4251.models.bert.modeling_bert_new import \
    BertForSequenceClassification as BFSCNew

glue_dataset_ls=["ax","cola","mnli","mnli_matched","mnli_mismatched",
                 "mrpc","qnli","qqp","rte","sst2","stsb","wnli"]

def train(model, optimizer, train_loader,val_loader,
          task,path,
          EPOCH,LR,DEVICE,
          batch_size=32,
          ):

    past_acc=-1.
    for epoch in range(EPOCH):
        correct = 0
        undetected = 0
        detected = 0

        print(f"-------EPOCH {epoch}-------------")
        for i,(inputs,attentions,typeids,labels) in enumerate(train_loader):
            inputs,attentions,typeids,labels=inputs.to(DEVICE),\
                attentions.to(DEVICE),typeids.to(DEVICE),labels.to(DEVICE)

            outputs = model(inputs,attentions,
                            typeids,labels=labels)

            prediction = torch.nn.functional.softmax(outputs.logits,dim=1)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predict_result = torch.nn.functional.softmax(outputs.logits, dim=1)
            predict_result = torch.argmax(predict_result, dim=1).cpu().numpy()

            if i%300==0:
                print(f"loss:{loss.item()}")
            
        acc=test(test_loader=val_loader,model=model,task=task,
                batch_size=batch_size,DEVICE=DEVICE)
        if acc>past_acc:
            model.save_pretrained(path)
            past_acc=acc


def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int,
                        required=False, help='训练的轮次')
    parser.add_argument('--lr', default=3e-4, type=float,
                        required=False, help='学习率')
    parser.add_argument('--cuda_num', default='6', type=str, required=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        required=False, help='训练batch size')
    parser.add_argument('--task', default="cola", type=str,
                        required=True,)
    parser.add_argument("--max_seq_length", default=128,
                        type=int, required=False, help="模型的最大输入长度")
    parser.add_argument("--train", default=1, type=int,
                        required=True, help="用以决定是训练模式还是测试模式")

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
    parser.add_argument('--root_dir', default='/home/liangzi/he_transformer/newglue/',
                        type=str, required=False,)
    return parser.parse_args()

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
    
    tmodel = AutoModelForSequenceClassification.from_pretrained(args.teach_ckpt)
    ttokenizer = AutoTokenizer.from_pretrained(args.teach_ckpt)

    smodel = BFSCNew.from_pretrained(args.stu_ckpt)
    stokenizer = AutoTokenizer.from_pretrained(args.stu_ckpt)
    tokenizer=ttokenizer

    optimizer = torch.optim.AdamW(smodel.parameters(), lr=LR)
    smodel = smodel.to(DEVICE)
    tmodel = tmodel.to(DEVICE)

    train_loader,val_loader,\
        test_loader = preprocess(tokenizer=tokenizer,device=DEVICE,
                                batch_size=BATCH_SIZE,task=task)

    if args.train==1:
        #============================================
        train(args,tmodel=tmodel,smodel=smodel, optimizer=optimizer,
              train_loader=train_loader,val_loader=val_loader,
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
    
    test(test_loader=val_loader,model=smodel,task=task,
         batch_size=BATCH_SIZE,DEVICE=DEVICE)

    print("Now on Original Student Model.")
    smodel=smodel.from_pretrained(args.stu_ckpt)
    smodel.to(DEVICE)
    smodel.eval()
    
    test(test_loader=val_loader,model=smodel,task=task,
         batch_size=BATCH_SIZE,DEVICE=DEVICE)
    

def train(args, tmodel, smodel, optimizer, train_loader,val_loader,
          task,
          EPOCH,LR,DEVICE,
          batch_size=32,
          ):
    kl_loss=torch.nn.KLDivLoss(reduction='batchmean')

    past_acc=-1.
    for epoch in range(EPOCH):
        correct = 0
        undetected = 0
        detected = 0

        print(f"-------EPOCH {epoch}-------------")
        for i,(inputs,attentions,typeids,labels) in enumerate(train_loader):
            inputs,attentions,typeids,labels=inputs.to(DEVICE),\
                attentions.to(DEVICE),typeids.to(DEVICE),labels.to(DEVICE)

            toutputs=tmodel(inputs,attentions,typeids,labels=labels,
                            output_hidden_states=True)
            teacher_logits=toutputs.logits

            outputs = smodel(inputs,attentions,
                             typeids,labels=labels,
                             output_hidden_states=True)

            entropy_loss=0.
            softlabel_loss=0.
            inter_loss=0.
            if args.using_entropy:
                entropy_loss = outputs.loss
            if args.using_softLabel:
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
            if args.using_interKL:
                ## we use MSE loss for that.
                mse_loss=0.
                lens=len(toutputs.hidden_states)
                for j in range(lens):
                    mse_loss+=F.mse_loss(toutputs.hidden_states[j],
                               outputs.hidden_states[j],reduction="mean")
                mse_loss/=lens
            
            a1,a2=0.33,0.33
            a3=1-a1-a2
            loss = a1*entropy_loss + a2*softlabel_loss + a3*inter_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predict_result = torch.nn.functional.softmax(outputs.logits, dim=1)
            predict_result = torch.argmax(predict_result, dim=1).cpu().numpy()

            if i%300==0:
                # print(f"loss:{loss.item()}")
                print(f"Loss:{loss}\tEntropy:{entropy_loss}\
                \tDistill:{softlabel_loss}\tInter:{inter_loss}")
            
        acc=test(test_loader=val_loader,model=smodel,task=task,
                batch_size=batch_size,DEVICE=DEVICE)
        if acc>past_acc:
            smodel.save_pretrained(args.stu_save_ckpt)
            past_acc=acc
    

## running entry
if __name__=="__main__":
    main1()
    print("EVERYTHING DONE.")


