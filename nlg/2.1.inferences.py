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
from trains1 import getFinetunedSet

def eval_vanilla_gpt2():
    model_path="./stage1_ckpts/GEM/web_nlg-epoch5-lr5e-05-bs1/"
    cuda_num=5
    infermodel=Inference(model_path,cuda_num)

    tr,va,te=getFinetunedSet(infermodel.tokenizer,
                             max_seq_length=infermodel.msl,
                             task="GEM/web_nlg",subset="en")
    # using validation dataset to test.
    seqls=[x[0] for x in va]
    newseqls=infermodel.inference(seqls)
    res=infermodel.evaluate(newseqls,seqls)
    print(res)



def main():
    eval_vanilla_gpt2()

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


