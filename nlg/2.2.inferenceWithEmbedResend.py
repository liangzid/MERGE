"""
======================================================================
2.2.INFERENCEWITHEMBEDRESEND --- 

RUNNING inference for embedding resend models.

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 16 二月 2023
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

from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
def eval_vanilla_gpt2():
    
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/"
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/___withConstantMatrixDistilled1114108e-50.01/"
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114003e-40.01/"
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01/"


    ## 这一版本即将达到可用范围，但还是差一点点。
    ### trainmodel 比validation的最好版本要好
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/"
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01/"

    # ## 待测试的新版本，在cosine loss的基础上又添加了mse loss
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01_testmseloss/"
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01_testmselosstrainmodel/"
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01_testmselossfinally/"

    # ## 待测试的新版本(无显著效果，放弃测试)
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01_tuneNoise/"

    # ## 待测试的新版本,在cosine和mse的基础上又添加了negative loss
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01_testNegativeLosstrainmodel/"


    ## 新版本，添加了dropout
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropout/"
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropouttrainmodel/"


    ### 该checkpoint是当下的最好结果。后续的所有修改计划基于该checkpoint进行
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropoutfinally/"


    # ## 在dropout之上添加了新一轮的retraining
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropoutfinally/ReTraining1114008e-50.01finally/"

    # 该方法追平了之前的结果
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropoutfinally/ReTraining1114008e-50.01trainmodel/"
    # model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropoutfinally/ReTraining1114008e-50.01/"
    model_path="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropoutfinally/ReTraining1114008e-50.01finally/"

    task="web_nlg"
    subset="release_v2"

    # task="e2e_nlg"
    # subtask=None

    cuda_num=0
    infermodel=Inference(model_path,cuda_num,
                         )

    bla_tokenizer = AutoTokenizer.from_pretrained(model_path)
    te=getTestDataSet(bla_tokenizer,split="test",
                             max_sentence_length=64,
                             task=task,subset=subset,withsep=True)
    dev=getTestDataSet(infermodel.tokenizer,split="dev",
                             max_sentence_length=infermodel.msl//2,
                             task=task,subset=subset,withsep=True)
    va,valabels=te
    # va,valabels=dev

    # using validation dataset to test.
    # seqls=[x[0] for x in va]
    seqls=va

    # seqls=seqls[:50]
    # valabels=valabels[:50]

    seqls=seqls[-50:]
    valabels=valabels[-50:]

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


