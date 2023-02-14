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
import time

import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM 
from transformers import pipeline
from transformers.generation.logits_process import  TemperatureLogitsWarper, RepetitionPenaltyLogitsProcessor,TopPLogitsWarper,TopKLogitsWarper,NoRepeatNGramLogitsProcessor,NoBadWordsLogitsProcessor

from transformersV4251.models.gpt2.gpt2_new import \
    GPT2LMHeadModel as BFSCNew
from transformersV4251.models.t5.modeling_t5 import \
    T5ForConditionalGeneration as T5New
from transformersV4251.models.bart.modeling_bart import \
    BartForConditionalGeneration as BartNew

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5ForConditionalGeneration
from transformers import BartForConditionalGeneration

# there might exist some bugs in the evaluate library
import evaluate
# as a replacement, we use nlg-metricverse as the evaluate metric.
from nlgmetricverse import NLGMetricverse


class Inference:
    def __init__(self,
                 model_path="data/helloworld",
                 cuda_num=6,
                 seed=3933, cuda=True,
                 approximation=False,
                 ):

        device = 'cuda:{}'.format(cuda_num) if cuda else 'cpu'
        self.device = device
        print('using device:{}'.format(device))
        # os.environ["CUDA_VISIBLE_DEVICES"] ="6"
        
        bla_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer = bla_tokenizer
        print("tokenizer loading done...")

        only_decoder=True
        if "gpt" in model_path:
            if not approximation:
                self.decoder=AutoModelForCausalLM\
                    .from_pretrained(model_path)
            else:
                self.decoder=BFSCNew\
                    .from_pretrained(model_path)
            
        elif "bart" in model_path or "t5" in model_path:
            only_decoder=False
            if "bart" in model_path:
                if not approximation:
                    self.decoder=BartForConditionalGeneration\
                        .from_pretrained(model_path)
                else:
                    self.decoder=BartNew\
                        .from_pretrained(model_path)
            elif "t5" in model_path:
                if not approximation:
                    self.decoder=T5ForConditionalGeneration\
                        .from_pretrained(model_path)
                else:
                    self.decoder=T5New\
                        .from_pretrained(model_path)
        else:
            self.decoder=AutoModelForCausalLM\
                .from_pretrained(model_path)

        self.only_decoder=only_decoder

        self.eos_token=self.tokenizer.eos_token
        self.eos_token_id=self.tokenizer.eos_token_id
        if only_decoder:
            self.sep_token=self.tokenizer.sep_token
        print("INFERENCE-MODEL-PATH: {}".format(model_path))

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

        repetition_penalty=2.5
        self.repetition_processor=RepetitionPenaltyLogitsProcessor(repetition_penalty)

        print(">> Waiting the NLG metrics loading...")
        t1=time.time()
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
        t2=time.time()
        print(f"time cost in load original metrics: {t2-t1}")

        # t1=time.time()
        # self.metricModels=NLGMetricverse(metrics=["bleu","rouge","meteor","chrf",
        #                                           "ter","bertscore","bleurt"])
        # t2=time.time()
        # print(f"time cost in load metrics: {t2-t1}")
        
    def inference(self, sequence, generate_mode_test="greedy"):
        new_sent = []
        progress=tqdm(total=len(sequence),desc="Inference Progress")
        print('==========starting testing==========')
        for seq in sequence:
            # print("input: {}".format(seq))
            # input_ids = self.tokenizer.encode(seq, return_tensors="pt")
            # print("sequence: ",seq)
            input_ids=seq
            # input_ids=seq.unsqueeze(0)
            input_ids = input_ids.to(self.device)
            try:
                if generate_mode_test == "greedy":
                    outputs=self.decoder.generate(input_ids=input_ids, max_length=self.max_target_length,                                 repetition_penalty=3.0,no_repeat_ngram_size=2,
                                                  pad_token_id=self.tokenizer.eos_token_id)
                else:
                    outputs=self.gen_embedResend(input_ids)

                sentence=self.tokenizer.decode(outputs[0],
                                skip_special_tokens=False)
                p=self.tokenizer.decode(seq[0],skip_special_tokens=False)
                # print("raw prefix: {}".format(p))
                # # print("raw prefix id: {}".format(seq))
                # print("raw gen sent: {}".format(sentence))
                if self.eos_token in sentence:
                    sentence=sentence.split(self.eos_token)[0]

                if self.only_decoder:
                    if self.sep_token in sentence:
                        sentence=sentence.split(self.sep_token)[-1]
                # print("post process sent: {}".format(sentence))

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
        one_refs=[x[0] for x in refs]
        big_res_dict={}
        for i,m in enumerate(self.metrics_ls):
            try:
                # if m=="bleurt":
                    # continue
                if m in self.multi_ref_ls:
                    big_res_dict[m]=self.metricsModel_ls[i]\
                                        .compute(predictions=hyps,
                                                    references=refs)
                else:
                    big_res_dict[m]=self.metricsModel_ls[i]\
                                        .compute(predictions=hyps,
                                                 references=one_refs)
            except Exception:
                big_res_dict[m]={"res":"empty, with error."}
                print("Error info:")
                print(Exception)
                print("------------------")
        return big_res_dict

    def evaluate2(self,hyps,refs):
        res=self.metricModels(predictions=hyps,references=refs)
        return res
        
    def gen_embedResend(self,prefix_ids):
        """
        Embedding resend style sentence generation.
        """
        # print(">>> USING EMBEDRESEND GENERATION.")

        # 1.2 then get the embeddings of ids.
        ## noted: here we only need the semantic embedding,
        # because the positional embedding can be added to, in models.
        embeddings=self.embedds(prefix_ids)
        # print(f"embeddings shape: {embeddings.shape}")
        if self.only_decoder:
            bs,sl,d=embeddings.shape
            gen_len=self.msl-sl
        else:
            bs,sl,d=embeddings.shape
            decoder_input_ids=self.tokenizer([" "],return_tensors="pt").input_ids
            decoder_input_ids=decoder_input_ids.to(self.device)
            decoder_input_embedds=self.embedds(decoder_input_ids)
            # print("decoder embedds shape: ",decoder_input_embedds.shape)
            sl=1
            gen_len=self.msl-sl
            

        first_token=0
        # 2 greedy forward generation
        with torch.no_grad():
            for _ in range(gen_len):
                first_token+=1
                if self.only_decoder:
                    output=self.decoder.forward(
                        inputs_embeds=embeddings,
                        output_hidden_states=True,
                        )
                else:
                    if first_token==1: # first input
                        output=self.decoder.forward(
                            prefix_ids,
                            decoder_inputs_embeds=decoder_input_embedds,
                            output_hidden_states=True,
                            )
                    else:
                        output=self.decoder.forward(
                            prefix_ids,
                            decoder_inputs_embeds=decoder_input_embedds,
                            output_hidden_states=True,
                            )
                if self.only_decoder:
                    decoder_input_ids=prefix_ids
                
                next_token_logits=output.logits[0,-1,:]
                next_token_distribution=F.softmax(next_token_logits,dim=-1)

                newdistribution=next_token_distribution

                # Appendix: repetition control
                newdistribution=self.repetition_processor(decoder_input_ids,
                                                          newdistribution.unsqueeze(0))
                # ## add temperature and subset sampling
                # newdistribution=self.temp_warper(decoder_input_ids,
                #                                    newdistribution)
                # newdistribution=self.top_p_warpper(decoder_input_ids,
                #                                    newdistribution)
                # newdistribution=self.temp_peak_warper(decoder_input_ids,
                #                                    newdistribution)
                # newdistribution=self.topk_warpper(decoder_input_ids,
                #                                   newdistribution)

                sorted_ids = torch.argsort(newdistribution.squeeze(0),
                                        dim=-1, descending=True)
                if self.only_decoder:
                    # here we just use the greedy search for generation
                    prefix_ids = torch.cat([prefix_ids,
                                    sorted_ids[None, 0, None]], dim=-1)
                    decoder_input_ids=prefix_ids
                else:
                    # here we just use the greedy search for generation
                    decoder_input_ids = torch.cat([decoder_input_ids,
                                    sorted_ids[None, 0, None]], dim=-1)

                # get new embeddings for next step's input.
                sl+=1
                if self.only_decoder:
                    embeddings=output.hidden_states[-1][:,:sl,:]
                else:
                    decoder_input_embedds=\
                        output.decoder_hidden_states[-1][:,:sl,:]

                # print(decoder_input_ids)
                if decoder_input_ids[0,-1]==self.eos_token_id:
                    break
        return decoder_input_ids

def main():
    inputt=["Aarhus | leader | Jacob_Bundsgaard<|sep|>"]
    inferenceModel=Inference(model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1t5-small",
                             cuda_num=7)

    inputt_id=inferenceModel.tokenizer(inputt,
                    return_tensors="pt").input_ids

    xxx=inferenceModel.inference([inputt_id],generate_mode_test="embedresend")
    print(xxx)

def main1_testEval():
    inferenceModel=Inference(model_path="./stage1_ckpts/web_nlg-epoch6-lr5e-05-bs1fianlly",
                    cuda_num=7)

    hyps=[]
    refs=[]
    res=inferenceModel.evaluate(hyps,refs)
    print(res)
    

if __name__=="__main__":
    main()
    # main1_testEval()

