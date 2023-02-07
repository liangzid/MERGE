"""
======================================================================
BERT_NEW_STRUCTURE ---

Python New Bert Implementation with only linear layer and ReLU.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created:  4 一月 2023
======================================================================
"""

import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch import matmul as mm
import time
from tqdm import tqdm

# from ...activations import ACT2FN
# from ...modeling_outputs import (
#     BaseModelOutputWithPastAndCrossAttentions,
#     BaseModelOutputWithPoolingAndCrossAttentions,
#     CausalLMOutputWithCrossAttentions,
#     MaskedLMOutput,
#     MultipleChoiceModelOutput,
#     NextSentencePredictorOutput,
#     QuestionAnsweringModelOutput,
#     SequenceClassifierOutput,
#     TokenClassifierOutput,
# )
# from ...modeling_utils import PreTrainedModel
# from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
# from ...utils import (
#     ModelOutput,
#     add_code_sample_docstrings,
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     logging,
#     replace_return_docstrings,
# )

# ------------------------ Code --------------------------------------

class BertFlattenForSeqCls(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = config.hidden_size
        self.inter_d=config.intermediate_size

        if hasattr(config,"msl"):
            self.msl=config.msl
        else:
            self.msl=128


        # ## vanilla version
        # self.M = nn.Parameter(torch.zeros(self.num_attention_heads,
        #                 self.msl,
        #                 self.msl))
        # self.Wvalue = nn.Linear(config.hidden_size, self.all_head_size)
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # self.activation=XXXX
        # self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        ## Linear Version
        self.M = nn.Parameter(torch.zeros(self.num_attention_heads,
                        self.msl,
                        self.msl))
        self.Wvalue = nn.Linear(config.hidden_size, self.all_head_size)
        self.dense0 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNormweight = nn.Parameter(torch.zeros(config.hidden_size))
        self.LayerNormbias = nn.Parameter(torch.zeros(config.hidden_size))
        self.denseFF0 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation=nn.ReLU()
        self.denseFF1 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNormFFweight = nn.Parameter(torch.zeros(config.hidden_size))
        self.LayerNormFFbias = nn.Parameter(torch.zeros(config.hidden_size))

        ## Linear Version
        self.M1 = nn.Parameter(torch.zeros(self.num_attention_heads,
                        self.msl,
                        self.msl))
        self.Wvalue1 = nn.Linear(config.hidden_size, self.all_head_size)
        self.dense01 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNormweight1 = nn.Parameter(torch.zeros(config.hidden_size))
        self.LayerNormbias1 = nn.Parameter(torch.zeros(config.hidden_size))
        self.denseFF01 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation1=nn.ReLU()
        self.denseFF11 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNormFFweight1 = nn.Parameter(torch.zeros(config.hidden_size))
        self.LayerNormFFbias1 = nn.Parameter(torch.zeros(config.hidden_size))

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        ## ========================= precomputed matrix
        self.init_d_flatten=torch.matmul(self.Wvalue.weight,self.dense0.weight)\
            *self.LayerNormweight
        self.init_d_flatten=torch.matmul(self.denseFF0.weight,
                                         self.init_d_flatten)
        self.init_d_bias_flatten=mm(self.denseFF0.weight,
                                              self.LayerNormbias) + \
            mm(self.denseFF0.weight,self.dense0.bias) +\
            self.denseFF0.bias

        self.init_M=self.M

        self.inter0_d_flatten=mm(self.denseFF1.weight.T,
                                 self.Wvalue1.weight)\
            *self.LayerNormFFweight

        self.inter0_d_flatten=torch.mm(self.inter0_d_flatten,
                                           self.dense01.weight,
                                           ).T

        self.inter0_d_flatten=torch.mul(self.inter0_d_flatten,
                                        self.LayerNormweight1.unsqueeze(1)\
                                        .expand(
            self.all_head_size,self.inter_d))
        self.inter0_d_flatten=torch.matmul(self.denseFF01.weight,
                                           self.inter0_d_flatten,
                                           )

        inter0_b_p1=mm(self.denseFF01.weight,mm(mm(self.denseFF1.bias,
                          self.Wvalue1.weight),
                          self.dense01.weight)*self.LayerNormFFweight1
                       )
        inter0_bp2=mm(self.denseFF01.weight,mm(self.Wvalue1.bias,
                          self.dense01.weight)*self.LayerNormFFweight1,
                       )
        inter0_bp3=mm(self.denseFF01.weight,self.dense01.bias\
                      *self.LayerNormFFweight1)
        inter0_bp4=self.denseFF01.bias
        inter0_b_p5=mm(self.LayerNormFFbias,
            torch.mm(self.Wvalue1.weight,
                     self.dense01.weight))*self.LayerNormFFweight1
        inter0_b_p5=mm(inter0_b_p5,self.denseFF01.weight.T)
        inter0_b_p6=mm(self.LayerNormbias1,self.denseFF01.weight.T)

        self.inter0_bias=inter0_b_p1+inter0_bp2+inter0_bp3+inter0_bp4+\
            inter0_b_p5+inter0_b_p6
        self.inter0_M=self.M1

        # print(self.denseFF11.weight.shape,self.LayerNormFFweight1.shape)
        self.final_d=self.denseFF11.weight*self.LayerNormFFweight1\
                                    .unsqueeze(1).expand(self.all_head_size,
                                                         self.inter_d)
        
        self.final_b=self.denseFF11.bias*self.LayerNormFFweight1.\
            unsqueeze(0).expand(-1,self.all_head_size)\
        +self.LayerNormFFbias1


    def forwardVanillaLinear(
        self,
        hidden_states: torch.Tensor,
    ):
        """
        shape of hidden_states: (bs,msl,d,1)

        """
        hidden_states=hidden_states.squeeze(-1)
        bs,msl,d=hidden_states.shape

        # shape of v : bs,msl,num_heads, feature_dim_per_head
        V=self.Wvalue(hidden_states).view(bs,msl,
                                          self.num_attention_heads,-1)
        V=torch.transpose(V,1,3)
        xo=torch.zeros(V.shape)
        for i_bs in range(bs):
            for i_shape in range(self.num_attention_heads):
                xo[i_bs,:,i_shape,:]=torch.matmul(V[i_bs,:,i_shape,:],
                                                  self.M[i_shape])
        xo=torch.transpose(xo,1,3).reshape(bs,msl,-1)
        # shape of xo : bs, msl, num_heads, feature_dim_per_head

        xo=self.dense0(xo)
        xo=xo*self.LayerNormweight+self.LayerNormbias
        xo=self.denseFF0(xo)
        xo=self.activation(xo)
        xo=self.denseFF1(xo)
        xo=xo*self.LayerNormFFweight+self.LayerNormFFbias

        # ================== first layer ends here ===================
        V=self.Wvalue1(xo).view(bs,msl,
                                          self.num_attention_heads,-1)
        V=torch.transpose(V,1,3)
        xo=torch.zeros(V.shape)
        for i_bs in range(bs):
            for i_shape in range(self.num_attention_heads):
                xo[i_bs,:,i_shape,:]=torch.matmul(V[i_bs,:,i_shape,:],
                                                  self.M1[i_shape])
        xo=torch.transpose(xo,1,3).reshape(bs,msl,-1)
        xo=self.dense01(xo)
        xo=xo*self.LayerNormweight1+self.LayerNormbias1
        xo=self.denseFF01(xo)
        xo=self.activation1(xo)
        xo=self.denseFF11(xo)
        xo=xo*self.LayerNormFFweight1+self.LayerNormFFbias1

        logits=self.classifier(xo[:,0,:])
        return logits

    def multiheadMut(self,x,M):
        bs,msl,d=x.shape
        x=x.view(bs,msl,self.num_attention_heads,-1)
        x=torch.transpose(x,1,3)
        xo=torch.zeros(x.shape)
        
        for i_bs in range(bs):
            for i_shape in range(self.num_attention_heads):
                xo[i_bs,:,i_shape,:]=torch.matmul(x[i_bs,:,i_shape,:],
                                    M[i_shape])
        xo=torch.transpose(xo,1,3).reshape(bs,msl,-1)
        return xo

    def forwardFastLinear(self,x):
        hidden_states=x.squeeze(-1)
        bs,msl,d=hidden_states.shape

        ## init transform
        xo=mm(x,self.init_d_flatten.T)
        xo=self.multiheadMut(xo,self.init_M)
        xo=xo+self.init_d_bias_flatten

        ## internal0 transform
        xo=mm(xo,self.inter0_d_flatten.T)
        xo=self.multiheadMut(xo,self.inter0_M)
        xo=xo+self.inter0_bias

        ## final transform
        xo=mm(xo,self.final_d.T)+self.final_b

        logits=self.classifier(xo[:,0,:])
        return logits 


    def forward(self,x,t=0):
        if t==0:
            return self.forwardVanillaLinear(x)
        else:
            return self.forwardFastLinear(x)

def main():
    from transformers import AutoConfig
    config=AutoConfig.from_pretrained("./save_models/saved_bert-base-uncased_taskcola-epoch30-lr3e-05-bs32")
    model=BertFlattenForSeqCls(config)
    inp=torch.ones((1,128,768))*1.5

    times=10000
    t1=time.time()
    for _ in tqdm(range(times)):
        outs=model.forward(inp,t=0)
    t2=time.time()
    print(outs,t2-t1)
    t1=time.time()
    for _ in tqdm(range(times)):
        outs=model.forward(inp,t=1)
    t2=time.time()
    print(outs,t2-t1)

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")
