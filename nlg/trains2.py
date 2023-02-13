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
from tqdm import tqdm

from datasets import load_dataset
from torch.utils.data import DataLoader

from transformersV4251.models.gpt2.gpt2_new import \
    GPT2LMHeadModel as BFSCNew

from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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
    parser.add_argument('--using_quadacti', default=0,
                        type=int, required=False,)
    parser.add_argument('--using_simLN', default=0,
                        type=int, required=False,)
    parser.add_argument('--root_dir', default='/home/liangzi/he_transformer/newglue/',
                        type=str, required=False,)
    parser.add_argument('--writer_dir',
                        type=str, required=False,default="./logs/1")
    parser.add_argument('--board_name',
                        type=str, required=True,)
    parser.add_argument('--max_grad_norm', default=1.0,
                        type=float, required=False,)
    return parser.parse_args()


def train(args, tmodel, smodel,
          optimizer, train_loader, val_loader,
          task,
          EPOCH,LR,DEVICE,
          batch_size=32,
          only_decoder=True,
          ):
    kl_loss=torch.nn.KLDivLoss(reduction='batchmean')
    tb_writer = SummaryWriter(log_dir=args.writer_dir+args.board_name)

    ii=0
    overall_step=0.
    tqdm1=tqdm(total=EPOCH)
    for epoch in range(EPOCH):
        ii+=1
        tqdm1.update(1)
        tqdm2=tqdm(total=len(train_loader))

        print(f"-------EPOCH {epoch}-------------")
        for i,x in enumerate(train_loader):
            if only_decoder:
                inps,atts=x
            else:
                inps,atts,outs=x
                outs=outs.to(DEVICE)

            overall_step+=1
            tqdm2.update(1)
            bs,msl=inps.shape
            inps,=inps.to(DEVICE),
            atts,=atts.to(DEVICE),

            if only_decoder:
                toutputs=tmodel(inps,attention_mask=atts,
                                labels=inps,
                                output_hidden_states=True)
                outputs = smodel(inps,attention_mask=atts,
                                labels=inps,
                                output_hidden_states=True)
            else:
                toutputs=tmodel(inps,attention_mask=atts,
                                decoder_input_ids=outs,
                                labels=outs,
                                output_hidden_states=True)
                outputs = smodel(inps,attention_mask=atts,
                                decoder_input_ids=outs,
                                labels=outs,
                                output_hidden_states=True)

            teacher_logits=toutputs.logits

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

                # print(f"teacher logitis: {teacher_logits.shape}")
                new_stu_logits=F.softmax(outputs.logits/args.tau,dim=2)
                # new_stu_logits+=1e-4
                new_tea_logits=F.softmax(teacher_logits/args.tau,dim=2)
                # print("stu logits:",new_stu_logits[0])

                if bs==1:
                    new_stu_logits=new_stu_logits.squeeze(0)
                    new_tea_logits=new_tea_logits.squeeze(0)
                    # print(new_stu_logits.shape,new_tea_logits.shape)
                    softlabel_loss=kl_loss(new_stu_logits.log(),
                                    new_tea_logits,
                                    )
                    # softlabel_loss/=msl
                else:
                    softlabel_loss=0.
                    for bsi in range(bs):
                        softlabel_loss+=kl_loss(new_stu_logits[bsi].log(),
                                                new_tea_logits[bsi],)

                # print("softlabelLoss",softlabel_loss)
                softlabel_loss*=args.tau**2

            if args.using_interKL==1:
                ## we use MSE loss for that.
                lens=len(toutputs.hidden_states)
                # print(f"length of hidden states layers: {lens}")
                for j in range(lens):
                    # print(toutputs.hidden_states[j])
                    # print(outputs.hidden_states[j])

                    # assert (toutputs.hidden_states[j]==\
                    #        outputs.hidden_states[j]).all()

                    templ=F.mse_loss(toutputs.hidden_states[j],
                               outputs.hidden_states[j],
                                reduction="mean")
                    # print(f"this part mse loss: {templ}")
                    inter_loss+=templ

                inter_loss/=lens

            # todo: frozon the lm head, to the embeddings
            if args.using_wordEmbedMSE==1:
                ## assert the 
                wordEmMSE_loss=0.
            
            # a1,a2,a3=0.25,0.25,0.25
            # a4=1-a1-a2-a3
            # loss = a1*entropy_loss + a2*softlabel_loss +\
            #     a3*inter_loss + a4*wordEmMSE_loss
            loss=0.33*entropy_loss+0.33*softlabel_loss+0.33*inter_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                smodel.parameters(), args.max_grad_norm)
            optimizer.step()

            if i%1==0:
                # print(f"loss:{loss.item()}")
                print(f"Loss:{loss}\tEntropy:{entropy_loss}\
                \tDistill:{softlabel_loss}\tInter:{inter_loss}\
                \twordEmbed:{wordEmMSE_loss}")

                tb_writer.add_scalar(args.board_name+"--LOSS",loss.item(),overall_step)
                tb_writer.add_scalar(args.board_name+"--SoftLabelOSS",softlabel_loss.item(),overall_step)
                tb_writer.add_scalar(args.board_name+"--CElOSS",entropy_loss.item(),overall_step)
                tb_writer.add_scalar(args.board_name+"--interLOSS",inter_loss,overall_step)


            if ii%100==0:
                print("Run Validating...")
                losses=test(test_loader=val_loader,
                         model=smodel,
                         task=task,
                         batch_size=batch_size,
                         DEVICE=DEVICE)
                tb_writer.add_scalar(args.board_name+"--valLOSS",
                                     losses,overall_step)

                if losses<past_losses:
                    smodel.save_pretrained(args.stu_save_ckpt)
                    past_losses=losses

    print("End Training.")

def main1():
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
    
    tmodel = AutoModelForCausalLM.from_pretrained(args.teach_ckpt)
    print("TEA Original embedding size: ",tmodel.get_input_embeddings().weight.shape[0])
    ttokenizer = AutoTokenizer.from_pretrained(args.teach_ckpt)
    tmodel.resize_token_embeddings(len(ttokenizer))

    config=AutoConfig.from_pretrained(args.stu_ckpt)
    if args.using_quadacti==1:
        config.activation_function="quad" # set to quad activation
    if args.using_simLN==1:
        config.layerNormType="sim" # set to quad activation
    smodel = BFSCNew.from_pretrained(args.stu_ckpt,config=config)
    print("STU Original embedding size: ",smodel.get_input_embeddings().weight.shape[0])
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

    optimizer = torch.optim.AdamW(smodel.parameters(), lr=LR)
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
    smodel=smodel.from_pretrained(args.stu_ckpt)
    smodel.to(DEVICE)
    smodel.eval()
    
    test(test_loader=teloader,model=smodel,task=task,
         batch_size=BATCH_SIZE,DEVICE=DEVICE)


## running entry
if __name__=="__main__":
    main1()
    print("EVERYTHING DONE.")


