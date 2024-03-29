"""
======================================================================
TRAIN_SLIDE --- 

windows slide embedding resend training.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 16 二月 2023
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
from transformersV4251.models.gpt2.gpt2_no_softmax import\
    GPT2LMHeadModel as GPT_nsm

from transformersV4251.models.t5.modeling_t5 import \
    T5ForConditionalGeneration as T5New
from transformersV4251.models.t5.t5_mpcformer import \
    T5ForConditionalGeneration as mpcT5

from transformersV4251.models.bart.new_bart import \
    BartForConditionalGeneration as BartNew
from transformersV4251.models.bart.bart_mpcformer import \
    BartForConditionalGeneration as mpcBart

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

def train(args, tmodel, smodel,prolayer,
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
    tqdm2=tqdm(total=args.train_step)
    past_losses=10000
    train_past_l=1e4
    for epoch in range(EPOCH):
        ii+=1
        tqdm1.update(1)
        if step_break==1:
            break
        print(f"-------EPOCH {epoch}-------------")
        for i,x in enumerate(train_loader):
            if overall_step>args.train_step:
                step_break=1
                break
            embedds=smodel.get_input_embeddings()
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
                emd_inps=embedds(inps)
                noise=(torch.rand(emd_inps.shape)-0.5)*2/(1/args.noise)
                # mask p% noise
                p=0.0
                mask_noise=torch.bernoulli(torch.ones_like(noise)*(1-p))
                noise=noise*mask_noise
                noise=noise.to(emd_inps.device)
                emd_inps=noise+emd_inps
                emd_inps=drop_layer(emd_inps)

                toutputs=tmodel(inps,
                                # attention_mask=atts,
                                labels=inps,
                                output_hidden_states=True)
                outputs = smodel(inputs_embeds=emd_inps,
                                # attention_mask=atts,
                                labels=inps,
                                output_hidden_states=True)
            else:
                #! bug finded: NO ER in this procedure.
                emd_outs=embedds(outs)
                noise=(torch.rand(emd_outs.shape)-0.5)*2/(1/args.noise)
                # mask p% noise
                p=0.0
                mask_noise=torch.bernoulli(torch.ones_like(noise)*(1-p))
                noise=noise*mask_noise
                noise=noise.to(emd_outs.device)
                emd_outs=noise+emd_outs
                emd_outs=drop_layer(emd_outs)

                toutputs=tmodel(inps,attention_mask=atts,
                                decoder_input_ids=outs,
                                labels=outs,
                                output_hidden_states=True)

                outputs = smodel(inps,attention_mask=atts,
                                decoder_inputs_embeds=emd_outs,
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
            # huberloss=nn.HuberLoss(reduction="mean",delta=0.01)
            # 1. first get the label embeddings. 
            if only_decoder:
                label_embedds=embedds(inps) # expect shape: bs,sl,d
            else:
                label_embedds=embedds(outs) # expect shape: bs,sl,d
            shift_laem=label_embedds[:,1:,:] 
            # 1.1. output hidden states
            if only_decoder:
                states=outputs.hidden_states[-1]
            else:
                states=outputs.decoder_hidden_states[-1]
            # print(states.shape) ## (bs,msl,d)
            if args.using_prolayer==1:
                states=prolayer(states)
            shift_sta=states[:,:-1,:]
            previous_sta=states[:,1:,:]

            if args.using_wordEmbedMSE==1:
                # shift_sta=prolayer(shift_sta)
                # 2. then calculate hte mse loss.
                wordEmMSE_loss+=F.mse_loss(shift_laem,shift_sta,
                                           reduction="mean")
                num_loss+=1
            if args.using_COSEm==1:
                labels=torch.ones((shift_sta.shape[0],
                                   shift_sta.shape[1])).to(shift_sta.device)
                # print("---")
                # print(shift_laem.shape)
                # print(shift_sta.shape)
                wordCos_loss=cosineEmbedLoss\
                    (shift_laem.reshape(-1,states.shape[-1]),
                              shift_sta.reshape(-1,states.shape[-1]),
                                              labels.reshape(-1))
                num_loss+=1
            if args.using_NEGAEm==1:
                labels=torch.ones((shift_sta.shape[0],
                                   shift_sta.shape[1])).to(shift_sta.device)*(-1)
                nega_loss=cosineEmbedLoss(shift_laem.reshape(-1,states.shape[-1]),
                                        previous_sta.reshape(-1,states.shape[-1]),
                                        labels.reshape(-1)
                                        )
                num_loss+=1

                ### in temp we do not use the huber loss.
                # huber_loss+=huberloss(shift_laem,shift_sta)

                # print(f"mse loss: {wordEmMSE_loss}\t cosine: {wordCos_loss}\t huber: {huber_loss}\t negative loss: {nega_loss}")

            
            # a1,a2,a3=0.25,0.25,0.25
            # a4=1-a1-a2-a3
            # loss = a1*entropy_loss + a2*softlabel_loss +\
            #     a3*inter_loss + a4*wordEmMSE_loss

            # print(f"total loss num: {total_num}")
            # loss=(wordEmMSE_loss+entropy_loss+softlabel_loss+inter_loss)/total_num
            # loss=(wordEmMSE_loss+entropy_loss)/2
            # loss=(wordEmMSE_loss+entropy_loss+wordEmMSE_loss1)/3
            # loss=(wordEmMSE_loss+entropy_loss+wordEmMSE_loss1-nega_loss)/total_num
            # loss=(wordEmMSE_loss+entropy_loss)/2
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


def main():
    args=setup_train_args()
    torch.autograd.set_detect_anomaly(True)

    # assert args.teach_ckpt!=args.stu_ckpt
    
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
        tmodel = T5ForConditionalGeneration.\
            from_pretrained(args.teach_ckpt)
    elif "bart" in args.teach_ckpt:
        tmodel = BartForConditionalGeneration.\
            from_pretrained(args.teach_ckpt)
    else:
        tmodel = AutoModelForCausalLM.from_pretrained(args.teach_ckpt)
    print("TEA Original embedding size: ",tmodel.get_input_embeddings().weight.shape[0])
    ttokenizer = AutoTokenizer.from_pretrained(args.teach_ckpt,
                            truncation="left" # left part truncation
                                               )
    tmodel.resize_token_embeddings(len(ttokenizer))

    config=AutoConfig.from_pretrained(args.teach_ckpt)
    if args.using_quadacti==1:
        config.activation_function="quad" # set to quad activation
    else:
        config.activation_function="gelu_new" # set to quad activation
    if args.using_simLN==1:
        config.layerNormType="sim" # set to quad activation
    else:
        config.layerNormType="no-sim" # set to quad activation
    if args.softmax2quad==1:
        # config.quad_softmax="quad"
        config.quad_softmax="relu"
        # config.quad_softmax="linear"
    else:
        config.quad_softmax="0"
    if args.no_res==1:
        print("no resiual connect using.")
        config.no_res="1"
    else:
        config.no_res="0"
        
    config.save_pretrained(args.stu_save_ckpt)

    print(f"config softmax2quad: {args.softmax2quad}")
    if args.softmax2quad==1:
        if 1==0:
            pass
        else:
            if "t5" in args.teach_ckpt:
                print(">>>>>Using MPCFORMER T5")
                smodel = mpcT5.from_pretrained(args.stu_ckpt,
                                    config=args.stu_save_ckpt)
            elif "bart" in args.teach_ckpt:
                print(">>>>>Using MPCFORMER bart")
                smodel = mpcBart.from_pretrained(args.stu_ckpt,
                                    config=args.stu_save_ckpt)
            else:
                print(">>>>>Using MPCFORMER gpt2")
                smodel = mpcGPT2.from_pretrained(args.stu_ckpt,
                                    config=args.stu_save_ckpt)
    elif "Constant" in args.stu_ckpt or args.using_quadacti==1 or args.using_simLN==1 or args.using_simLN==2:
        print("Using new structure.")
        if "t5" in args.teach_ckpt:
            smodel = T5New.\
                from_pretrained(args.stu_ckpt,config=args.stu_save_ckpt)
        elif "bart" in args.teach_ckpt:
            smodel = BartNew.\
                from_pretrained(args.stu_ckpt,config=args.stu_save_ckpt)
        else:
            smodel = BFSCNew.from_pretrained(args.stu_ckpt,config=args.stu_save_ckpt)

    elif (args.using_simLN==0 and args.using_quadacti==0) or "WithEm" in args.stu_save_ckpt:
        print("Using vanilla structure.")
        if "t5" in args.teach_ckpt:
            smodel = T5ForConditionalGeneration.\
                from_pretrained(args.stu_ckpt)
        elif "bart" in args.teach_ckpt:
            smodel = BartForConditionalGeneration.\
                from_pretrained(args.stu_ckpt)
        else:
            smodel = AutoModelForCausalLM.from_pretrained(args.stu_ckpt)
        
    print("STU Original embedding size: ",smodel.get_input_embeddings().weight.shape[0])

    if args.no_softmax==1 and "t5" not in args.teach_ckpt \
       and "bart" not in args.teach_ckpt:
        smodel = GPT_nsm.from_pretrained(args.stu_ckpt,
                                         config=args.stu_save_ckpt)
        print("Using Constant Version without Softmax Functions.")

    # print(smodel.transformer.h[2].attn.M)
    

    tokenizer=ttokenizer
    stokenizer=tokenizer
    tokenizer.save_pretrained(args.stu_save_ckpt)
    tokenizer.save_pretrained(args.stu_save_ckpt+"trainmodel")
    smodel.resize_token_embeddings(len(tokenizer))
    print("length of vocab in tokenizer: ",len(tokenizer))

    print("whether the last linear map has grad: ",
            smodel.lm_head.weight.requires_grad)

    ## add new layer
    # from projectModel import ProjecLayer
    # prolayer=ProjecLayer(config.n_embd,config.n_embd)
    prolayer=None

    # for name,param in smodel.named_parameters():
    #     if "M" in name:
    #         print("find Attn Matrix,\
    #         now set the grad to false.")
    #         param.required_grad=False
    # print("whether the Constant Attn has grad: ",
    #         False)

    optimizer1 = torch.optim.AdamW(smodel.parameters(), lr=LR,
                                  weight_decay=args.weight_decay,)

    # optimizer2 = torch.optim.AdamW(prolayer.parameters(), lr=LR,
    #                               weight_decay=args.weight_decay,)
    optimizer2=None
    smodel = smodel.to(DEVICE)
    # prolayer = prolayer.to(DEVICE)
    prolayer = None
    tmodel = tmodel.to(DEVICE)

    print(f"max sequence length: {args.max_seq_length}")
    trs,vas,tes=getFinetunedSet(tokenizer,args.max_seq_length,
                                task,subtask,only_decoder)
    
    if task=="mutiwoz_nlg":
        vas=Subset(vas, np.arange(1000))

    trloader=DataLoader(trs,batch_size=batch_size,
                            shuffle=True,drop_last=False)
    valoader=DataLoader(vas,batch_size=batch_size,
                            shuffle=True,drop_last=True)
    teloader=DataLoader(tes,batch_size=batch_size,
                            shuffle=True,drop_last=True)

    #============================================
    if args.train==1:
        if args.softmax2quad==1:
            from train_baseline_distill import vanilla_distill as vd
            vd(args,tmodel=tmodel,smodel=smodel,prolayer=prolayer,
                optimizer1=optimizer1,
                optimizer2=optimizer2,
                train_loader=trloader,val_loader=valoader,
                task=task,
                batch_size=BATCH_SIZE,
                EPOCH=EPOCH,LR=LR,
                DEVICE=DEVICE,tokenizer=tokenizer,
                only_decoder=only_decoder)
            
        else:
            train(args,tmodel=tmodel,smodel=smodel,prolayer=prolayer,
                optimizer1=optimizer1,
                optimizer2=optimizer2,
                train_loader=trloader,val_loader=valoader,
                task=task,
                batch_size=BATCH_SIZE,
                EPOCH=EPOCH,LR=LR,
                DEVICE=DEVICE,tokenizer=tokenizer,
                only_decoder=only_decoder)
        smodel.save_pretrained(args.stu_save_ckpt+"finally")

        if args.using_prolayer==1:
            torch.save(prolayer.state_dict(),
                       args.stu_save_ckpt+"finally_prolayer.pt")
        tokenizer.save_pretrained(args.stu_save_ckpt)
        tokenizer.save_pretrained(args.stu_save_ckpt+"finally")
        tokenizer.save_pretrained(args.stu_save_ckpt+"差不多")
    #============================================

    smodel=smodel.from_pretrained(args.stu_save_ckpt)
    if args.using_prolayer==1:
        prolayer=torch.load(args.stu_save_ckpt+"_prolayer.pt",
                            map_location="cpu")
        prolayer.to(DEVICE)
        prolayer.eval()
    
    # test(test_loader=val_loader,model=smodel,task=task,
    #      batch_size=BATCH_SIZE,DEVICE=DEVICE)

    print("Now on Original Student Model.")
    # smodel=smodel.from_pretrained(args.stu_save_ckpt+"finally")
    smodel=smodel.from_pretrained(args.stu_save_ckpt+"trainmodel")
    smodel.to(DEVICE)
    smodel.eval()
    
    res=test(test_loader=teloader,model=smodel,task=task,
             batch_size=BATCH_SIZE,DEVICE=DEVICE,
             only_decoder=only_decoder)
    print(res)


## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")
