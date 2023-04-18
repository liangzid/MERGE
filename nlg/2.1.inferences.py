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
from trains1 import getFinetunedSet,getTestDataSet

def eval_vanilla_gpt2():
    
    withsep=True

    # model_path="./stage1_ckpts/e2e_nlg-epoch3-lr5e-05-bs1gpt2/"
    # model_path="./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs1gpt2/"
    model_path="./stage1_ckpts/multiwoz_nlg-epoch6-lr5e-5-bs32bart-base/addQuad1000104019e-50.010.60.70.75finally/"

    if "bart" in model_path or "t5" in model_path:
        withsep=False

    task="multiwoz_nlg"
    subset=None
    gentype="ER"

    cuda_num=5

    infermodel=Inference(model_path,cuda_num,
                         approximation=False,
                         use_filter=0,
                         )

    te=getTestDataSet(infermodel.tokenizer,split="test",
                             max_sentence_length=infermodel.msl//2,
                             task=task,subset=subset,withsep=True)

    va,valabels=te
    # va,valabels=dev

    seqls=va

    # seqls=seqls[:50]
    # valabels=valabels[:50]

    if gentype=="vanilla":

        # # # print(seqls[0])
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
        newseqls=infermodel.inference(seqls,generate_mode_test="embedResend")

        if subset is None:
            genpath=model_path+task+"embedresend.json"
        else:
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


