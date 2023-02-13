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

def eval_vanilla_gpt2():
    
    model_path="./stage1_ckpts/web_nlg-epoch6-lr5e-05-bs1fianlly/"
    model_path="./stage1_ckpts/web_nlg-epoch1-lr5e-05-bs1/"

    task="web_nlg"
    subset="release_v2"

    cuda_num=5
    infermodel=Inference(model_path,cuda_num)

    tr,va,te=getTestDataSet(infermodel.tokenizer,
                             max_sentence_length=infermodel.msl//2,
                             task=task,subset=subset)
    va,valabels=te
    # using validation dataset to test.
    # seqls=[x[0] for x in va]
    seqls=va
    seqls=seqls[:100]
    valabels=valabels[:100]

    # print(seqls[0])
    newseqls=infermodel.inference(seqls)
    res=infermodel.evaluate(newseqls,valabels)
    print("----Vanilla Greedy Search Results----")
    print(res)
    # newseqls=infermodel.inference(seqls)
    # res=infermodel.evaluate(newseqls,valabels)
    # print("----Embedding Resend Results----")
    # print(res)


def main():
    eval_vanilla_gpt2()

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


