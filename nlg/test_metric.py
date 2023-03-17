"""
======================================================================
TEST_METRIC ---

BLEURT, COMET, and others.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 16 三月 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

from evaluate import load

def main1():
    pres=["hello world","how do you do","hehe"]
    refs=["hello word","how do u do","hehehehe"]

    # BLEURT
    bleurt=load("bleurt",module_type="metric")
    results=bleurt.compute(predictions=pres,references=refs)
    print(f"results: {results}")

    # COMET


## running entry
if __name__=="__main__":
    # main()
    main1()
    print("EVERYTHING DONE.")


