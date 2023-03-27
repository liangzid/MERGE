"""
======================================================================
2.13.QUAD_INFER ---

approximation and ER methods.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created:  6 三月 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

from inference import Inference
from trains1 import getFinetunedSet,getTestDataSet
import json
from collections import OrderedDict

def main():

    # ## 1. e2e nlg
    # task="e2e_nlg"
    # subset=None

    # ## 2. dailydialog
    # task="daily_dialog"
    # subset=None

    ## 3. multiwoz 2.1 nlg
    task="multiwoz_nlg"
    subset=None
    withsep=True

    # ## 4. web nlg
    # task="web_nlg"
    # subset="release_v2"

    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4gpt2/"
    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4t5-small/"
    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4bart-base/"

    # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/addQuad1000104108e-50.010.40.7finally/"
    # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/addQuad1000104108e-50.010.40.7trainmodel/"
    # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/addQuad1000104108e-50.010.40.7epoch3/"

    # ## with interkl loss
    # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/addQuad1010104108e-50.010.40.7finally/"

    # ## only ER
    # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/DropoutTraining1004008e-50.01finally"


    # ## with simLN
    # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/addQuad1000104118e-50.010.40.7finally/"

    # ## with constant matrix
    # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/___withConstantMatrixaddQuad1000104108e-50.010.40.7finally/"

    # # ## weighted loss
    # # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/addQuad1000104118e-50.010.40.70.9finally/"
    # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/addQuad1000104118e-50.010.40.70.75finally/"

    ## 4
    # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/addQuad1000104008e-50.010.40.70.75finally/"
    ## 3
    # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/others_1000104008e-50.010.40.70.5finally/"
    ## 6
    # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/others_1000104118e-50.010.40.70.5finally/"

    ## ER+CM+simLN
    model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/addQuad1000104118e-50.010.40.70.75finally/"
    # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/___withConstantMatrixOnlyCM1000104018e-50.010.40.70.75epoch2/"
    

    gentype="ER"
    # gentype="vanilla"


    ## ---------------------------------------------
    cuda_num=7
    infermodel=Inference(model_path,cuda_num,
                         # approximation=True,
                         approximation=False,
                         use_filter=1,
                         )

    te=getTestDataSet(infermodel.tokenizer,split="test",
                             max_sentence_length=infermodel.msl//2,
                             task=task,subset=subset,withsep=withsep)

    va,valabels=te
    seqls=va

    # seqls=seqls[:100]
    # valabels=valabels[:100]


    if gentype=="vanilla":

        # print(seqls[0])
        newseqls=infermodel.inference(seqls)

        if subset is None:
            genpath=model_path+task+"greedy.json"
        else:
            genpath=model_path+task+subset+"greedy.json"

        with open(genpath, 'w',encoding='utf8') as f:
            json.dump([newseqls,valabels],f,ensure_ascii=False,indent=4)
        print("data save done.")

        with open(genpath, 'r',encoding='utf8') as f:
            data=json.load(f,object_pairs_hook=OrderedDict)
        newseqls,valabels=data

        print(valabels[0])
        res=infermodel.evaluate(newseqls,valabels)
        print("----Vanilla Greedy Search Results----")
        print(res)

    else:
        # newseqls=infermodel.inference(seqls,generate_mode_test="embedResend")

        if subset is None:
            genpath=model_path+task+"embedresend.json"
        else:
            genpath=model_path+task+subset+"embedresend.json"

        # with open(genpath, 'w',encoding='utf8') as f:
        #     json.dump([newseqls,valabels],f,ensure_ascii=False,indent=4)
        # print("res save done.")

        # from collections import OrderedDict
        with open(genpath, 'r',encoding='utf8') as f:
            data=json.load(f,object_pairs_hook=OrderedDict)
        newseqls,valabels=data
        res=infermodel.evaluate(newseqls,valabels)
        print("----Embedding Resend Results----")
        print(res)


## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


