"""
======================================================================
2.12.OTHERTASKS_INFERENCE ---

Inference for other tasks.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created:  1 三月 2023
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

    # ## 3. multiwoz 2.1 nlg
    task="multiwoz_nlg"
    subset=None

    # ## 4. web nlg
    # task="web_nlg"
    # subset="release_v2"

    # ## # 5. common gen
    # task="common_gen"
    # subset=None

    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4gpt2/"
    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4t5-small/"
    model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4gpt2/"
    withsep=False
    withsep=True
    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4bart-base/"

    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4bart-base/"
    
    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs4gpt2/"
    # model_path=f"./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/DropoutTraining1114008e-50.01finally/"
    # model_path=f"./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/DropoutTraining1114008e-50.01trainmodell/"

    # # multiwoz ckpt
    # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/"
    # model_path=f"./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/DropoutTraining1004008e-50.01finally"

    # # web nlg ckpt
    # model_path=f"./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/DropoutTraining1004008e-50.01trainmodel/"
    # model_path=f"./stage1_ckpts/{task}-epoch3-lr5e-05-bs1gpt2/"

    gentype="ER"
    # gentype="vanilla"

    ## ---------------------------------------------
    cuda_num=5
    infermodel=Inference(model_path,cuda_num,
                         # approximation=True
                         )

    if task=="common_gen":
        te=getTestDataSet(infermodel.tokenizer,split="validation",
                                max_sentence_length=infermodel.msl//2,
                                task=task,subset=subset,withsep=withsep)
        va,valabels=te
    else:
        te=getTestDataSet(infermodel.tokenizer,split="test",
                                max_sentence_length=infermodel.msl//2,
                                task=task,subset=subset,withsep=withsep)
        va,valabels=te
    seqls=va

    if gentype=="vanilla":

        # print(seqls[0])
        # newseqls=infermodel.inference(seqls)

        if subset is None:
            genpath=model_path+task+"greedy.json"
        else:
            genpath=model_path+task+subset+"greedy.json"

        # with open(genpath, 'w',encoding='utf8') as f:
        #     json.dump([newseqls,valabels],f,ensure_ascii=False,indent=4)
        # print("data save done.")

        with open(genpath, 'r',encoding='utf8') as f:
            data=json.load(f,object_pairs_hook=OrderedDict)
        newseqls,valabels=data

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

