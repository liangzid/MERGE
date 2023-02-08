"""
======================================================================
ENCRYP_ONLY_FOR_TEST ---

Only for test the Linear layer, to compare the inference time between

matrix multiplication and the linear layers.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created:  7 二月 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

import math
import time

import torch
import torch.nn.functional as F
from torch import matmul as mm

import crypten
import crypten.nn as cnn
import crypten.nn as nn 
import crypten.communicator as comm

from utils import softmax_2RELU, activation_quad

from models import Bert,BertEmbeddings,BertLayer,BertAttention,BertSelfAttention,BertSelfOutput,BertIntermediate,BertOutput


class BertTest(nn.Module):
    def __init__(self,config,timing):
        super(BertTest,self).__init__()
        self.config=config
        self.timing=timing
        # self.config.num_labels=2
        self.device=torch.device("cuda:0")

        self.embeddings=BertEmbeddings(config,timing)

        # self.last_w=torch.ones((config.hidden_size,config.hidden_size)).to(self.device)
        # self.last_b=torch.ones((config.hidden_size)).to(self.device)

        self.layer=nn.Linear(config.hidden_size,config.hidden_size)
        
        # self.activation=activation_quad()
        self.activation=nn.ReLU()

    def forward(self,x):
        # print(x.shape)
        xo=self.embeddings(x)

        ## final transform
        # xo=xo.matmul(self.last_w.T)+self.last_b
        xo=self.layer(xo)

        return xo

    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0



## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


