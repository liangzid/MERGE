"""
======================================================================
INFERENCE ---

Generation procedure of NLG models.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 10 二月 2023
======================================================================
"""


# ------------------------ Code --------------------------------------
from pprint import pprint as ppp
from typing import List,Tuple,Dict
import json
import random
import argparse
from tqdm import tqdm
import logging
import os
from os.path import join, exists
from itertools import zip_longest, chain
from datetime import datetime
import pickle

import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM 
from transformers import pipeline

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn as nn

import evaluate

class Inference:
    def __init__(self,
                 model_path="data/helloworld",
                 cuda_num=6,
                 seed=3933, cuda=True,
                 ):

        device = 'cuda:{}'.format(cuda_num) if cuda else 'cpu'
        self.device = device
        print('using device:{}'.format(device))
        # os.environ["CUDA_VISIBLE_DEVICES"] ="6"
        
        bla_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer = bla_tokenizer
        print("tokenizer loading done...")
        self.eos_token=self.tokenizer.eos_token
        self.sep_token=self.tokenizer.sep_token
        print("INFERENCE-MODEL-PATH: {}".format(model_path))
        self.decoder=AutoModelForCausalLM\
            .from_pretrained(model_path)
        self.decoder.resize_token_embeddings(len(bla_tokenizer))
        self.embedds=self.decoder.get_input_embeddings()
        print("model loading done...")

        self.decoder.to(device).eval()
        self.embedds.to(device).eval()

        multi_gpu = False
        # if args.cuda and torch.cuda.device_count() > 1:
        #     LOGGER.info("Let's use GPUs to train")    
        #     decoder = DataParallel(decoder, device_ids=[int(i) for i in args.device.split(',')])
        #     multi_gpu = True

        num_parameters = 0
        parameters = self.decoder.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        print('number of all parameters: {}'\
              .format(num_parameters/1e6))

        self.max_target_length=128
        self.msl=self.max_target_length

        ## calculate the running examples.
        self.metrics_ls=["bleu","meteor","chrf","ter",
                                  "bertscore","bleurt",
                   "nist_mt","meteor","rouge"]
        self.multi_ref_ls=["bleu","ter","nist_mt"]

        
        self.metricsModel_ls=[]
        for metric in self.metrics_ls:
            if metric=="bleurt":
                self.metricsModel_ls.append(evaluate.load(metric,
                                    module_type="metric"))
            elif metric=="chrf":
                self.metricsModel_ls.append(evaluate.load(metric,
                                    word_order=2))
            else:
                self.metricsModel_ls.append(evaluate.load(metric))
        
    def inference(self, sequence, generate_mode_test="greedy"):
        new_sent = []
        progress=tqdm(total=len(sequence),desc="Inference Progress")
        print('==========starting testing==========')
        for seq in sequence:
            # print("input: {}".format(seq))
            # input_ids = self.tokenizer.encode(seq, return_tensors="pt")
            input_ids=seq.unsqueeze(0)
            input_ids = input_ids.to(self.device)
            try:
                if generate_mode_test == "greedy":
                    outputs=self.decoder.generate(input_ids=input_ids, max_length=self.max_target_length,                                 repetition_penalty=2.5,no_repeat_ngram_size=3,
                                                  pad_token_id=self.tokenizer.eos_token_id)
                else:
                    outputs=self.gen_embedResend(input_ids)

                sentence=self.tokenizer.decode(outputs[0],
                                skip_special_tokens=False)
                p=self.tokenizer.decode(seq,skip_special_tokens=False)
                print("raw prefix: {}".format(p))
                print("raw prefix id: {}".format(seq))
                print("raw gen sent: {}".format(sentence))
                if self.eos_token in sentence:
                    sentence=sentence.split(self.eos_token)[0]
                if self.sep_token in sentence:
                    sentence=sentence.split(self.sep_token)[1]
                print("post process sent: {}".format(sentence))

                new_sent.append(sentence)

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    print("WARNING: ran out of memory,\
                    times: {}".format(oom_time))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    print(str(exception))
                    raise exception
            progress.update(1)
        print('=============testing finished====================')
        return new_sent

    def evaluate(self,hyps,refs):
        """
        Both `hyps` and `refs` are one-array lists.
        """
        refss=[[x] for x in refs]
        big_res_dict={}
        for i,m in enumerate(self.metrics_ls):
            print(f"metrics: {m}")
            try:
                if m in self.multi_ref_ls:
                    big_res_dict[m]=self.metricsModel_ls[i]\
                                        .compute(predictions=hyps,
                                                    references=refss)
                else:
                    big_res_dict[m]=self.metricsModel_ls[i]\
                                        .compute(predictions=hyps,
                                                 references=refs)
            except:
                big_res_dict[m]={"res":"empty, with error."}
        return big_res_dict
        
    def gen_embedResend(self,prefix_ids):
        """
        Embedding resend style sentence generation.
        """

        # 1.2 then get the embeddings of ids.
        ## noted: here we only need the semantic embedding,
        # because the positional embedding can be added to, in models.
        embeddings=self.embedds(prefix_ids)
        print(f"embeddings shape: {embeddings.shape}")
        bs,sl,d=embeddings.shape

        # 2 greedy forward generation
        with torch.no_grad():
            for _ in range(self.msl):
                output=self.decoder.forward(
                    inputs_embeds=embeddings,
                    output_hidden_states=True,
                    )
                
                next_token_logits=output.logits[0,-1,:]
                next_token_distribution=F.softmax(next_token_logits,dim=-1)
                newdistribution=next_token_distribution
                sorted_ids = torch.argsort(newdistribution.squeeze(0),
                                        dim=-1, descending=True)
                # here we just use the greedy search for generation
                decoder_input_ids = torch.cat([decoder_input_ids,
                                sorted_ids[None, 0, None]], dim=-1)

                # get new embeddings for next step's input.
                sl+=1
                embeddings=output.hidden_states[-1][:,:sl,:]

                # print(decoder_input_ids)
                if decoder_input_ids[0,-1]==self.eos_token_id:
                    break
        return decoder_input_ids

def main():
    inputt="Aarhus | leader | Jacob_Bundsgaard<|sep|>"
    inferenceModel=Inference(model_path="./stage1_ckpts/GEM/web_nlg-epoch3-lr5e-05-bs1/",cuda_num=7)

    inputt_id=inferenceModel.tokenizer(inputt)

    xxx=inferenceModel.inference([inputt_id[0]])
    print(xxx)

if __name__=="__main__":
    main()

