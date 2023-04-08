"""
======================================================================
2.15.VARY_TASK_367METHODS ---

new vary task inference experiments.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 29 三月 2023
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

    ## 2. dailydialog
    task="e2e_nlg"
    subset=None

    # ## 2. dailydialog
    # task="daily_dialog"
    # subset=None

    # ## 5. common gen
    # task="common_gen"
    # subset=None

    # ## 3. multiwoz 2.1 nlg
    # task="multiwoz_nlg"
    # subset=None
    # # withsep=False

    # ## 4. web nlg
    # task="web_nlg"
    # subset="release_v2"

    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4gpt2/"
    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4t5-small/"
    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4bart-base/"


    withsep=True
    # # 3
    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4gpt2/_stuaddQuad1000104008e-50.010.40.70.75finally/"
    # cuda_num=5


    # # # 6
    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4gpt2/_stuaddQuad1000104118e-50.010.40.70.5finally/"
    # cuda_num=6

    # ## 7
    model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs1gpt2/_stuaddQuad1000104118e-50.010.40.70.75finally/"

    ## -------------- trash-------------------------------------------------
    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4gpt2/_stuaddQuad1000104113e-50.010.40.70.75trainmodel/"
    # model_path=f"./stage1_ckpts/common_gen-epoch3-lr5e-05-bs4gpt2/_stuTestCommonGen1000104118e-50.010.40.250.5finally/"
    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4gpt2/_stuaddQuad1000104118e-50.010.40.70.75epoch1/"
    # model_path=f"./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/_stuaddQuad1000104118e-50.010.40.70.75finally/"
    # model_path=f"./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/_stuaddQuad1000104118e-50.010.40.20.25finally/"
    ## -------------- trash-------------------------------------------------
    cuda_num=6
    

    gentype="ER"
    # gentype="vanilla"

    ## ---------------------------------------------
    infermodel=Inference(model_path,cuda_num,
                         # approximation=True,
                         approximation=False,
                         use_filter=1,
                         )

    if task=="common_gen":
        te=getTestDataSet(infermodel.tokenizer,split="validation",
                                max_sentence_length=infermodel.msl//2,
                                task=task,subset=subset,withsep=withsep)
    else:
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


