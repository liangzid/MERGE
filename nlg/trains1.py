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
        res=tokenizer(train_t,padding="longest",
                      truncation=True,
                    max_length=1024,return_tensors="pt")
        dset=TensorDataset(res.input_ids,
                        )
        return dset
    return getSet("train"),getSet("validation"),getSet("test")

def multiwoz_se(split):
    PATH=f"/home/liangzi/datasets/multi_woz_21_nlg/{split}.json"
    from collections import OrderedDict
    with open(PATH, 'r',encoding='utf8') as f:
        data=json.load(f,object_pairs_hook=OrderedDict)
    inps,outs=data["acts"],data["resps"]
    sinps=[]
    
    for x in inps:
        segs=[", ".join(act[:3]) for act in x]
        seg="; ".join(segs)
        sinps.append(seg)
        # print(seg)

    newouts=[]
    for x in outs:
        newouts.append(x)
    outs=newouts

    # print(len(sinps),len(outs))
    return sinps,outs
            
def daily_dialog(split):
    PATH=f"/home/liangzi/datasets/daily_dialog_post/{split}.json"
    from collections import OrderedDict
    with open(PATH, 'r',encoding='utf8') as f:
        data=json.load(f,object_pairs_hook=OrderedDict)
    inps,outs=data["inps"],data["resps"]

    return inps,outs

def getFinetunedSet(tokenizer,
                    max_sentence_length=256,
                    task="web_nlg",subset="release_v2",
                    only_decoder=True):
    """
    For Downstream Tasks based on Conditional Generation.
    task and subtask enums:
    e2e_nlg, subset:none
    """
    sep_token="<|sep|>"
    sep_token=tokenizer.sep_token
    if task =="daily_dialog":
        sep_token=""
    eos_token=tokenizer.eos_token
    if only_decoder:
        bos_token=""
    else:
        bos_token=tokenizer.bos_token

    def getSet(split="train"):
        if task in ["web_nlg","e2e_nlg","GEM/taskmaster","common_gen"]:
            if subset is not None:
                train_set=load_dataset(task,subset,split=split)
            else:
                train_set=load_dataset(task,split=split)
        elif task == "multiwoz_nlg":
            train_set=multiwoz_se(split)
        elif task == "daily_dialog":
            train_set=daily_dialog(split)

        inps=[]
        outs=[]
        if "web_nlg" in task:
            for x in train_set:
                # print("data: ",x)
                inps.append(" ; ".join(x["modified_triple_sets"]\
                                       ["mtriple_set"][0]))
                outs.append(x["lex"]["text"][0])
        elif "common_gen" in task:
            for x in train_set:
                inps.append(", ".join(x["concepts"]))
                outs.append(x["target"])
        elif "e2e_nlg" in task:
            for x in train_set:
                inps.append(x["meaning_representation"])
                outs.append(x["human_reference"])
        elif "multiwoz_nlg" in task or "daily_dialog" in task:
            inps,outs=train_set
            train_set=inps
        elif "taskmaster" in task:
            for x in train_set:
                inps.append(x["context"])
                outs.append(x["target"])
        
        if only_decoder:
            outs=[inps[i]+sep_token+outs[i]+eos_token\
                for i in range(len(inps))]

            outss=tokenizer(outs,padding="longest",
                            truncation=True,
                        max_length=max_sentence_length,return_tensors="pt")
            dset=TensorDataset(outss.input_ids,outss.attention_mask,
                            )
        else:
            inps=tokenizer(inps,padding="longest",truncation=True,
                           max_length=max_sentence_length,
                           return_tensors="pt")
            outs=[bos_token+x+eos_token for x in outs]
            outs=tokenizer(outs,padding="longest",truncation=True,
                           max_length=max_sentence_length,
                           return_tensors="pt")
            dset=TensorDataset(inps.input_ids,
                               inps.attention_mask,
                               outs.input_ids,
                            )
        return dset

    if "web_nlg" in task:
        names=["train","dev","test"]
    elif task in ["e2e_nlg","taskmaster","daily_dialog","common_gen"]:
        names=["train","validation","test"]
    elif "multiwoz_nlg" in task:
        names=["train","val","test"]
    return getSet(names[0]),getSet(names[1]),getSet(names[2])

def getTestDataSet(tokenizer,split="test",
                    max_sentence_length=128,
                    task="web_nlg",subset="release_v2",withsep=False):
    """
    For Downstream Tasks based on Conditional Generation.
    task and subtask enums:
    + GEM/web_nlg
        + en
        + ru
    + e2e_nlg, subset:none
    """
    if withsep:
        sep_token="<|sep|>"
    else:
        sep_token=""
    if task=="daily_dialog":
        sep_token=""
    # sep_token=tokenizer.sep_token
    eos_token=tokenizer.eos_token

    def getSet(split="train"):
        if task in ["web_nlg", "e2e_nlg", "GEM/taskmaster", "common_gen"]:
            if subset is not None:
                train_set=load_dataset(task,subset,split=split)
            else:
                train_set=load_dataset(task,split=split)
        elif task=="multiwoz_nlg":
            train_set=multiwoz_se(split)
        elif task=="daily_dialog":
            train_set=daily_dialog(split)

        inps=[]
        outs=[]
        if "web_nlg" in task:
            for x in train_set:
                inps.append(" ; ".join(x["modified_triple_sets"]\
                                       ["mtriple_set"][0]))
                outs.append(x["lex"]["text"])
        elif "e2e_nlg" in task:
            for x in train_set:
                inps.append(x["meaning_representation"])
                outs.append(x["human_reference"])
        elif "common_gen" in task:
            for x in train_set:
                inps.append(", ".join(x["concepts"]))
                outs.append(x["target"])
        elif "multiwoz_nlg" in task or "daily_dialog" in task:
            inps,outs=train_set
            train_set=inps
        elif "taskmaster" in task:
            for x in train_set:
                inps.append(x['context'])
                outs.append(x['target'])

        # merge the same inputs.
        inpout_dict={}
        for i, inp in enumerate(inps):
            out=outs[i]
            if inp not in inpout_dict:
                inpout_dict[inp]=[out]
            else:
                inpout_dict[inp].append(out)
        inps=[]
        outs=[]
        for k,v in inpout_dict.items():
            inps.append(k)
            outs.append(v)
            
        labels=outs
        outs=inps

        prefix_id_ls=[]
        for text in outs:
            ou=tokenizer(text+sep_token,padding="longest",
                        truncation=True,
                        max_length=max_sentence_length,
                            return_tensors="pt")
            # print(ou.input_ids)
            # print(type(ou.input_ids))
            if withsep==False:
                x=ou.input_ids
            elif ou.input_ids[0][-1]==50257 or task=="daily_dialog":
                x=ou.input_ids
            else:
                x=torch.cat((ou.input_ids[0],torch.tensor([50257])),0).unsqueeze(0)
            # print(x)
                
            # print("ou shape: ",ou)
            prefix_id_ls.append(x)

        return prefix_id_ls,labels
    # if "web_nlg" in task:
    #     names=["train","dev","test"]
    # elif "e2e_nlg" in task:
    #     names=["train","validation","test"]
    # return getSet(names[0]),getSet(names[1]),getSet(names[2])
    return getSet(split)
    
def trainConditional(model,
          optimizer,
          train_loader,
          val_loader,
          test_loader,
          task,
          save_path,
          EPOCH,LR,DEVICE,
        tokenizer,
          batch_size=32,
        only_decoder=True,
          ):

    tb_writer = SummaryWriter(log_dir=f"./logs/stage1train/{task}")
    board_name=f"{task}_"
    ii=0
    past_losses=10000
    tqdm1=tqdm(total=EPOCH)
    eos_token_id=tokenizer.eos_token_id
    loss_func=CrossEntropyLoss(ignore_index=eos_token_id)
    for epoch in range(EPOCH):
        tqdm1.update(1)
        tqdm2=tqdm(total=len(train_loader))

        print(f"-------EPOCH {epoch}-------------")
        for i,x in enumerate(train_loader):
            if only_decoder:
                inps,atts=x
            else:
                inps,atts,outs=x
                outs=outs.to(DEVICE)
            # print("----------------")
            # print(tokenizer.decode(inps[0]))
            # print(tokenizer.decode(outs[0]))

            tqdm2.update(1)
            # print(ii)
            ii+=1
            inps,=inps.to(DEVICE),
            atts,=atts.to(DEVICE),
            # print(tokenizer.decode(inps[0]))

            if only_decoder:
                outputs = model(inps,
                                attention_mask=atts,
                                labels=inps)
            else:
                # todo: fix the encoder decoder writing.
                outputs = model(inps,
                                attention_mask=atts,
                                # decoder_input_ids=outs,
                                labels=outs)

            loss = outputs.loss
            logits=outputs.logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tb_writer.add_scalar(board_name+"loss",loss.item(),ii)

            if ii%300==0:
                print(f"loss:{loss.item()}")

            if ii%1000==0:
                print("Run Validating...")
                losses=test(test_loader=val_loader,
                         model=model,
                         task=task,
                         batch_size=batch_size,
                            DEVICE=DEVICE,
                        only_decoder=only_decoder
                            )
                print(f">>Val Loss: {losses}")
                tb_writer.add_scalar(board_name+"valloss",
                                     losses.item(),ii)

                # lossess=test(test_loader=test_loader,
                #          model=model,
                #          task=task,
                #          batch_size=batch_size,
                #              DEVICE=DEVICE,
                #         only_decoder=only_decoder
                #              )
                # print(f">>Test Loss: {lossess}")
                # tb_writer.add_scalar(board_name+"testloss",
                #                      lossess.item(),ii)

                if losses<past_losses:
                    print(" -->now save a better model.")
                    print(f"in epoch {epoch}, step {i}.")

                    tokenizer.save_pretrained(save_path)
                    model.save_pretrained(save_path)
                    past_losses=losses

    print("End Training.")
            
def test(test_loader,model,task,batch_size=32,
         DEVICE="cpu",only_decoder=True):
    model.eval()
    losses=0.

    with torch.no_grad():
        for i,x in enumerate(test_loader):
            if only_decoder:
                inps,atts=x
            else:
                inps,atts,outs=x
                outs=outs.to(DEVICE)

            inps,=inps.to(DEVICE),
            atts,=atts.to(DEVICE),

            if only_decoder:
                outputs = model(inps,
                                attention_mask=atts,
                                labels=inps)
            else:
                outputs = model(inps,
                                attention_mask=atts,
                                # decoder_input_ids=outs,
                                labels=outs)
            loss = outputs.loss
            # print(loss)
            losses+=loss

    losses/=((i+1)*batch_size)
        
    model.train()
    return losses

def testNew(test_loader,model,task,batch_size=32,
         DEVICE="cpu",only_decoder=True):
    """
    calculate the loss based on self-defined forward function.
    """
    model.eval()
    losses=0.

    loss_func=CrossEntropyLoss(reduction="mean",
                                )
    with torch.no_grad():
        for i,x in enumerate(test_loader):
            if only_decoder:
                inps,atts=x
            else:
                inps,atts,outs=x
                outs=outs.to(DEVICE)

            inps,=inps.to(DEVICE),
            atts,=atts.to(DEVICE),

            if only_decoder:
                outputs = model(inps,
                                attention_mask=atts,
                                labels=inps)
            else:
                outputs = model(inps,
                                attention_mask=atts,
                                decoder_input_ids=outs,
                                labels=outs)
            logits=outputs.logits[:,:,:]
            bs,msl,v=logits.shape
            # print(f"logits shape: {logits.shape}")
            # distri=F.softmax(logits,dim=-1)
            distri=logits

            if only_decoder:
                loss1=loss_func(distri[:,:-1,:].reshape(-1,v),
                            inps[:,1:].reshape(-1))
            else:
                loss1=loss_func(distri[:,:-1,:].reshape(-1,v),
                            outs[:,1:].reshape(-1))
            loss=loss1
            losses+=loss
    losses/=(i+1)
    model.train()
    return losses

def main():
    EPOCH = 3
    # LR = 5e-5 
    LR = 5e-5 
    DEVICE = torch.device("cuda:5")
    # DEVICE = torch.device("cpu")
    BATCH_SIZE =32
    batch_size=BATCH_SIZE
    task_ls=["web_nlg","e2e_nlg"]
    subtaskls=["release_v2",None]

    # task="web_nlg"
    # subtask="release_v2"

    # task="e2e_nlg"
    # subtask=None

    # task="multiwoz_nlg"
    # subtask=None

    # task="daily_dialog"
    # subtask=None

    task="common_gen"
    subtask=None

    prefix_path="/home/liangzi/models/"

    model_name="gpt2/"
    # model_name="t5-small/"
    # model_name="bart-base/"
    print(model_name)
    frmpth=prefix_path+model_name

    if "gpt" in frmpth:
        only_decoder=True
    elif "t5" in frmpth or "bart" in frmpth:
        only_decoder=False
    else:
        only_decoder=True
    print(f"The Backbone is Only a Decoder: {only_decoder}.")
    
    PATH = f'./stage1_ckpts/{task}-epoch{EPOCH}-lr{LR}-bs{BATCH_SIZE}{model_name}'

    # model = BFSCNew.from_pretrained(frmpth)
    if only_decoder:
        model = AutoModelForCausalLM.from_pretrained(frmpth)
    else:
        if "t5" in frmpth:
            model = T5ForConditionalGeneration.from_pretrained(frmpth)
        elif "bart" in frmpth:
            config=AutoConfig.from_pretrained(frmpth)
            config.decoder_start_token_id=config.bos_token_id
            model = BartForConditionalGeneration.from_pretrained(frmpth)
        else:
            model = AutoModelForCausalLM.from_pretrained(frmpth)
            
    tokenizer = AutoTokenizer.from_pretrained(frmpth,
                                              truncation="left" # left part truncation
                                              )
    if "t5" in frmpth:
        tokenizer.bos_token="<pad>"
        tokenizer.sep_token=""
        tokenizer.pad_token=tokenizer.eos_token
    elif "bart" in frmpth:
        tokenizer.bos_token="<s>"
        tokenizer.sep_token=""
        # tokenizer.pad_token=tokenizer.eos_token
    else:
        tokenizer.pad_token=tokenizer.eos_token

    if only_decoder:
        tokenizer.add_tokens(["<|sep|>",],special_tokens=True)
        tokenizer.sep_token="<|sep|>"
    model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model = model.to(DEVICE)

    trs,vas,tes=getFinetunedSet(tokenizer,128,task,
                    subtask,only_decoder)
    # truncation the val set for fast training.
    vas=Subset(vas, np.arange(1000))

    print(f"train set len: {len(trs)}")
    print(f"validation set len: {len(vas)}")
    print(f"test set len: {len(tes)}")


    print(f"batch_size: {batch_size}")
    trloader=DataLoader(trs,batch_size=batch_size,
                            shuffle=True,drop_last=False)
    valoader=DataLoader(vas,batch_size=batch_size,
                            shuffle=False,drop_last=True)
    teloader=DataLoader(tes,batch_size=batch_size,
                            shuffle=False,drop_last=True)

    tokenizer.save_pretrained(PATH+"fianlly")
    #============================================
    trainConditional(model, optimizer,
                     trloader,valoader,teloader,
                     task,
                     PATH,
                     batch_size=BATCH_SIZE,
                     EPOCH=EPOCH,LR=LR,
                     DEVICE=DEVICE,tokenizer=tokenizer,
                     only_decoder=only_decoder)
    tokenizer.save_pretrained(PATH+"fianlly")
    model.save_pretrained(PATH+"fianlly")
    #============================================

    model=model.from_pretrained(PATH)
    model.to(DEVICE)
    model.eval()
    
    # test(test_loader=val_loader,model=model,task=task,
    #      batch_size=BATCH_SIZE,DEVICE=DEVICE)

    res=test(test_loader=valoader,model=model,task=task,
             batch_size=BATCH_SIZE,DEVICE=DEVICE,
             only_decoder=only_decoder)
    print(res)

    res=test(test_loader=teloader,model=model,task=task,
             batch_size=BATCH_SIZE,DEVICE=DEVICE,
             only_decoder=only_decoder)
    print(res)


## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


