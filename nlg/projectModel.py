"""
======================================================================
PROJECTMODEL ---

Model for mapping `hidden states` into `embeddings`.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 16 二月 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
# from torchvision.transforms import Compose as ComposeTransformation
# import tensorboardX


class ProjecLayer(nn.Module):
    def __init__(self,d_from,d_to):
        super(ProjecLayer,self).__init__()
        self.model=nn.Linear(d_from,d_to)

        self.model.weight.data.normal_(mean=0.0,
                                       std=0.02)
        self.model.bias.data.zero_()

    def forward(self,x):
        return self.model(x)


## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


