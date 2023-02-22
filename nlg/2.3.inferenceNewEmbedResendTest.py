"""
======================================================================
2.3.INFERENCENEWEMBEDRESENDTEST ---

New version test. Go!

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 20 二月 2023
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

def eval_gpt2():
    
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/"

    ## 新版本，添加了dropout
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropout/"
    model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropouttrainmodel/"

    # add extra linear projection
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropouttrainmodel/WithLinearProjectiontrainmodel"
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropouttrainmodel/WithLinearProjection"
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropouttrainmodel/WithLinearProjectionfinally"
    model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropouttrainmodel/WithLinearProjection"
    

    task="web_nlg"
    subset="release_v2"

    # task="e2e_nlg"
    # subtask=None

    cuda_num=1
    infermodel=Inference(model_path,cuda_num,have_project=True
                         )

    te=getTestDataSet(infermodel.tokenizer,split="test",
                             max_sentence_length=infermodel.msl//2,
                             task=task,subset=subset,withsep=True)
    dev=getTestDataSet(infermodel.tokenizer,split="dev",
                             max_sentence_length=infermodel.msl//2,
                             task=task,subset=subset,withsep=True)
    va,valabels=te
    # va,valabels=dev

    # using validation dataset to test.
    # seqls=[x[0] for x in va]
    seqls=va

    seqls=seqls[:50]
    valabels=valabels[:50]

    # print(seqls[0])
    newseqls=infermodel.inference(seqls)
    genpath=model_path+task+subset+"greedy.json"
    with open(genpath, 'w',encoding='utf8') as f:
        json.dump([newseqls,valabels],f,ensure_ascii=False,indent=4)
    print("data save done.")
    # from collections import OrderedDict
    with open(genpath, 'r',encoding='utf8') as f:
        data=json.load(f,object_pairs_hook=OrderedDict)
    newseqls,valabels=data

    res=infermodel.evaluate(newseqls,valabels)
    print("----Vanilla Greedy Search Results----")
    print(res)

    # print(newseqls)
    # print(valabels)
    # res=infermodel.evaluate2(newseqls,valabels)
    # print(res)

    # newseqls=infermodel.inference(seqls,generate_mode_test="embedResend")
    # genpath=model_path+task+subset+"embedresend.json"
    # with open(genpath, 'w',encoding='utf8') as f:
    #     json.dump([newseqls,valabels],f,ensure_ascii=False,indent=4)
    # print("res save done.")

    # # from collections import OrderedDict
    # with open(genpath, 'r',encoding='utf8') as f:
    #     data=json.load(f,object_pairs_hook=OrderedDict)
    # newseqls,valabels=data
    # res=infermodel.evaluate(newseqls,valabels)
    # print("----Embedding Resend Results----")
    # print(res)


## running entry
if __name__=="__main__":
    eval_gpt2()
    print("EVERYTHING DONE.")


