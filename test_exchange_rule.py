# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp
import torch

def test1():
    a=torch.rand((1,4,10))
    wd1=torch.rand((10,10))
    wd2=torch.rand((10,5))
    wm=torch.rand((4,4))

    res1=torch.matmul(a,wd1)
    res1=torch.matmul(res1.transpose(1,2),wm).transpose(1,2)
    res1=torch.matmul(res1,wd2)

    res2=torch.matmul(a.transpose(1,2),wm).transpose(1,2)
    res2=torch.matmul(res2,wd1)
    res2=torch.matmul(res2,wd2)

    print(res1)
    print(res2)

## running entry
if __name__=="__main__":
    test1()
    print("EVERYTHING DONE.")


