"""
======================================================================
TRAIN_BASELINE_DISTILL ---

Distill model training for existing baselines that do not need to resend
the embeddings.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 13 四月 2023
======================================================================
"""

# ------------------------ Code --------------------------------------
from tqdm import tqdm
## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from transformersV4251.models.gpt2.gpt2_new import \
    GPT2LMHeadModel as BFSCNew
from transformersV4251.models.gpt2.gpt2_mpcformer import \
    GPT2LMHeadModel as mpcGPT2
from transformersV4251.models.t5.modeling_t5 import \
    T5ForConditionalGeneration as T5New
from transformersV4251.models.bart.new_bart import \
    BartForConditionalGeneration as BartNew

from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM
from transformers import T5ForConditionalGeneration
from transformers import BartForConditionalGeneration
from transformers import AutoTokenizer
from transformers import AutoConfig
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import tensor

import json
import numpy as np
import argparse

from trains1 import getFinetunedSet,test,testNew
from trains2 import main1,setup_train_args

def vanilla_distill(args, tmodel, smodel,prolayer,
          optimizer1,optimizer2, train_loader, val_loader,
          task,
          EPOCH,LR,DEVICE,tokenizer,
          batch_size=32,
          only_decoder=True,
          ):
    kl_loss=torch.nn.KLDivLoss(reduction='batchmean')
    loss_func=CrossEntropyLoss(reduction="none")
    drop_layer=nn.Dropout(p=args.dropout_rate)
    tb_writer = SummaryWriter(log_dir=args.writer_dir+args.board_name)
    no_save_差不多model=True
    ii=0
    overall_step=0.
    step_break=0
    tqdm1=tqdm(total=EPOCH)
    past_losses=10000
    train_past_l=1e4
    for epoch in range(EPOCH):
        ii+=1
        tqdm1.update(1)
        tqdm2=tqdm(total=args.train_step)
        if step_break==1:
            break
        print(f"-------EPOCH {epoch}-------------")
        for i,x in enumerate(train_loader):
            if overall_step>args.train_step:
                step_break=1
                break
            # embedds=smodel.get_input_embeddings()
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

                toutputs=tmodel(inps,
                                # attention_mask=atts,
                                labels=inps,
                                output_hidden_states=True)
                outputs = smodel(inps,
                                # attention_mask=atts,
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

            num_loss=0
            entropy_loss=0.
            softlabel_loss=0.
            inter_loss=0.
            wordEmMSE_loss=0.
            if args.using_entropy==1:
                entropy_loss1 = outputs.loss
                distri=outputs.logits # we cannot use SOFTMAX!
                if only_decoder:
                    entropy_loss=loss_func(distri[:,:-1,:].reshape(bs*(msl-1),-1),
                                        inps[:,1:].reshape(-1))
                else:
                    entropy_loss=loss_func(distri[:,:-1,:].reshape(-1,
                                                            distri.size(-1)),
                                        outs[:,1:].reshape(-1))
                entropy_loss=entropy_loss.reshape(bs,-1)

                # weights=torch.linspace(1.,0.,steps=msl-1).to(DEVICE)
                # entropy_loss=torch.matmul(entropy_loss,weights)
                entropy_loss=torch.mean(entropy_loss)
                num_loss+=1
                
            if args.using_softLabel==1:
                new_stu_logits=F.softmax(outputs.logits/args.tau,dim=2)
                new_tea_logits=F.softmax(teacher_logits/args.tau,dim=2)

                if bs==1:
                    new_stu_logits=new_stu_logits.squeeze(0)
                    new_tea_logits=new_tea_logits.squeeze(0)
                    softlabel_loss=kl_loss(new_stu_logits.log(),
                                    new_tea_logits,
                                    )
                else:
                    softlabel_loss=0.
                    for bsi in range(bs):
                        softlabel_loss+=kl_loss(new_stu_logits[bsi].log(),
                                                new_tea_logits[bsi],)

                softlabel_loss*=args.tau**2
                num_loss+=1

            if args.using_interKL==1:
                ## we use MSE loss for that.
                lens=len(toutputs.hidden_states)
                # print(f"length of hidden states layers: {lens}")
                # print(toutputs.hidden_states)
                # print(outputs.hidden_states)
                for j in range(lens-1):
                    if j> (lens-1)//2:
                        break
                    templ=F.mse_loss(toutputs.hidden_states[j],
                               outputs.hidden_states[j],
                                reduction="mean")
                    # print(f"this part mse loss: {templ}")
                    inter_loss+=templ

                inter_loss/=(lens-1)
                num_loss+=1

            wordEmMSE_loss=0.
            wordCos_loss=0.
            huber_loss=0.
            nega_loss=0.
            cosineEmbedLoss=nn.CosineEmbeddingLoss(reduction="mean")
            if args.lamda==0.5:
                loss=(entropy_loss + softlabel_loss + inter_loss \
                    +wordEmMSE_loss +wordCos_loss+nega_loss)/num_loss
            else:
                loss=(entropy_loss + softlabel_loss + inter_loss \
                    +wordEmMSE_loss+nega_loss)/(num_loss-1)*(1-args.lamda) + \
                    args.lamda*wordCos_loss

            if loss<train_past_l and i%100==0:
                print("SaveNewTrainModel")
                smodel.save_pretrained(args.stu_save_ckpt+"trainmodel")
                # torch.save(prolayer.state_dict(),
                #            args.stu_save_ckpt+"trainmodel_prolayer.pt")
                train_past_l=loss

            optimizer1.zero_grad()
            # optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                smodel.parameters(), args.max_grad_norm)
            optimizer1.step()
            # optimizer2.step()

            if i%1==0:
                # print(f"loss:{loss.item()}")
                print(f"Loss:{loss}  Entropy:{entropy_loss}\
                  Distill:{softlabel_loss}  Inter:{inter_loss}\
                  EmMSE:{wordEmMSE_loss} Emcos:{wordCos_loss} EmNega: {nega_loss}")

                tb_writer.add_scalar(args.board_name+"--LOSS",loss.item(),overall_step)
                # tb_writer.add_scalar(args.board_name+"--SoftLabelOSS",softlabel_loss.item(),overall_step)
                tb_writer.add_scalar(args.board_name+"--CElOSS",entropy_loss.item(),overall_step)
                # tb_writer.add_scalar(args.board_name+"--interLOSS",inter_loss,overall_step)

            if overall_step%300==0:
                smodel.save_pretrained(args.stu_save_ckpt+"finally")
                tokenizer.save_pretrained(args.stu_save_ckpt+"finally")
                if args.using_prolayer==1:
                    torch.save(prolayer.state_dict(),
                            args.stu_save_ckpt+"finally_prolayer.pt")
                print("Run Validating...")
                losses=testNew(test_loader=val_loader,
                         model=smodel,
                         task=task,
                         batch_size=batch_size,
                            DEVICE=DEVICE,
                            only_decoder=only_decoder)
                tb_writer.add_scalar(args.board_name+"--valLOSS",
                                     losses,overall_step)
                print(f">>>VAL loss: {losses}")

                if losses<past_losses:
                    print(f"find a better model, in epoch {epoch} step {i}")
                    smodel.save_pretrained(args.stu_save_ckpt)
                    if args.using_prolayer==1:
                        torch.save(prolayer.state_dict(),
                                args.stu_save_ckpt+"_prolayer.pt")
                    past_losses=losses
                if losses<0.25 and no_save_差不多model:
                    smodel.save_pretrained(args.stu_save_ckpt+"差不多")
                    if args.using_prolayer==1:
                        torch.save(prolayer.state_dict(),
                                args.stu_save_ckpt+"差不多_prolayer.pt")
                    no_save_差不多model=False
        # epoch level
        smodel.save_pretrained(args.stu_save_ckpt+f"epoch{epoch}")
        tokenizer.save_pretrained(args.stu_save_ckpt+f"epoch{epoch}")
        if args.using_prolayer==1:
            torch.save(prolayer.state_dict(),
                    args.stu_save_ckpt+f"epoch{epoch}_prolayer.pt")

    print("End Training.")





