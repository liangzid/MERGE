"""
======================================================================
2.1.INFERENCES ---

Inference Experiments.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 10 二月 2023
======================================================================
"""


# ------------------------ Code --------------------------------------
from inference import Inference
from trains1 import getFinetunedSet,getTestDataSet
import json
from collections import OrderedDict

def eval_vanilla_gpt2():
    
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/"
    model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1t5-small/"
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1bart-base/"
    # model_path="./stage1_ckpts/e2e_nlg-epoch3-lr5e-05-bs1gpt2/"
    # model_path="./stage1_ckpts/e2e_nlg-epoch3-lr5e-05-bs1t5-small/"
    # model_path="./stage1_ckpts/e2e_nlg-epoch3-lr5e-05-bs1bart-base/"

    task="web_nlg"
    subset="release_v2"

    # task="e2e_nlg"
    # subset=None

    cuda_num=4
    infermodel=Inference(model_path,cuda_num)

    te=getTestDataSet(infermodel.tokenizer,split="test",
                             max_sentence_length=infermodel.msl//2,
                             task=task,subset=subset,withsep=False)
    va,valabels=te
    # using validation dataset to test.
    # seqls=[x[0] for x in va]
    seqls=va

    # seqls=seqls[:3]
    # valabels=valabels[:3]

    # # print(seqls[0])
    # newseqls=infermodel.inference(seqls)
    # genpath=model_path+task+subset+"greedy.json"
    # with open(genpath, 'w',encoding='utf8') as f:
    #     json.dump([newseqls,valabels],f,ensure_ascii=False,indent=4)
    # print("data save done.")
    # # from collections import OrderedDict
    # with open(genpath, 'r',encoding='utf8') as f:
    #     data=json.load(f,object_pairs_hook=OrderedDict)
    # newseqls,valabels=data

    # res=infermodel.evaluate(newseqls,valabels)
    # print("----Vanilla Greedy Search Results----")
    # print(res)

    # print(newseqls)
    # print(valabels)
    # res=infermodel.evaluate2(newseqls,valabels)
    # print(res)

    newseqls=infermodel.inference(seqls,generate_mode_test="embedResend")
    genpath=model_path+task+subset+"embedresend.json"
    with open(genpath, 'w',encoding='utf8') as f:
        json.dump([newseqls,valabels],f,ensure_ascii=False,indent=4)
    print("res save done.")

    # from collections import OrderedDict
    with open(genpath, 'r',encoding='utf8') as f:
        data=json.load(f,object_pairs_hook=OrderedDict)
    newseqls,valabels=data
    res=infermodel.evaluate(newseqls,valabels)
    print("----Embedding Resend Results----")
    print(res)

def main():
    eval_vanilla_gpt2()

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


