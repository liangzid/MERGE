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
from crypten.nn import Parameter
import crypten.communicator as comm

from utils import softmax_2RELU, activation_quad

from models import Bert,BertEmbeddings,BertLayer,BertAttention,BertSelfAttention,BertSelfOutput,BertIntermediate,BertOutput


class BertTest(nn.Module):
    def __init__(self,config,timing):
        super(BertTest,self).__init__()
        self.config=config
        self.timing=timing
        # self.config.num_labels=2
        self.device=torch.device(config.device)

        self.embeddings=BertEmbeddings(config,timing)
        self.embeddings.cuda()

        # self.last_w=torch.ones((config.hidden_size,config.hidden_size)).to(self.device)
        # self.last_b=torch.ones((config.hidden_size)).to(self.device)
        self.moduleList=cnn.ModuleList([
            nn.Linear(config.hidden_size,
                        config.hidden_size).to(self.device)\
            for _ in range(12)
            ])
        
        self.w2ls=[]
        self.b2ls=[]
        for _ in range(12):
            self.w2ls.append(torch.ones((config.hidden_size,
                        config.hidden_size)).to(self.device))
            self.b2ls.append(torch.ones((config.hidden_size)).\
                             to(self.device))
        for i in range(12):
            self.w2ls[i]=crypten.cryptensor(self.w2ls[i],src=0)
        for i in range(12):
            self.b2ls[i]=crypten.cryptensor(self.b2ls[i],src=0)
        
        # self.activation=activation_quad()
        self.activation=nn.ReLU()

    def forward1(self,x):
        # print(x.shape)
        xo=self.embeddings(x)

        ## final transform
        # xo=xo.matmul(self.last_w.T)+self.last_b
        for i in range(12):
            t0=time.time()
            c0=comm.get().get_communication_stats()

            xo=self.moduleList[i](xo)

            c1=comm.get().get_communication_stats()
            t1=time.time()

            self.timing["LinearTime"]+=(t1-t0)
            self.timing["LinearCommTime"]+=(c1['time']-c0['time'])
            self.timing["LinearCommByte"]+=(c1['bytes']-c0['bytes'])

        return xo

    def forward(self,x):
        # print(x.shape)
        xo=self.embeddings(x)

        ## final transform
        # xo=xo.matmul(self.last_w.T)+self.last_b
        for i in range(12):
            t0=time.time()
            c0=comm.get().get_communication_stats()
            xo=xo.matmul(self.w2ls[i])+self.b2ls[i]
            c1=comm.get().get_communication_stats()
            t1=time.time()

            self.timing["LinearTime"]+=(t1-t0)
            self.timing["LinearCommTime"]+=(c1['time']-c0['time'])
            self.timing["LinearCommByte"]+=(c1['bytes']-c0['bytes'])

        return xo

    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0



## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


