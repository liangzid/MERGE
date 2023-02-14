"""
======================================================================
NEWLAYERNORM ---

element-wise multiplication for Layer Normalization

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 13 二月 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

import torch
import torch.nn as nn


class SimpleLayerNorm(nn.Module):
    def __init__(self,d):
        super(SimpleLayerNorm,self).__init__()
        self.weight=nn.Parameter(torch.tensor((d),dtype=torch.float))
        self.bias=nn.Parameter(torch.tensor((d),dtype=torch.float))

    def forward(self,x):
        x=x*self.weight+self.bias
        return x


## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


