"""
======================================================================
3.1.EVAL_RES_BYREAD --- 

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 15 四月 2023
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
    withsep=True

    # ## 1. multiwoz bart vanilla
    # task="multiwoz_nlg"
    # subset=None
    # model_path="./stage1_ckpts/multiwoz_nlg-epoch6-lr5e-5-bs32bart-base/"
    # gentype="vanilla"

    # ## 2. multiwoz bart ER
    # task="multiwoz_nlg"
    # subset=None
    # model_path="./stage1_ckpts/multiwoz_nlg-epoch6-lr5e-5-bs32bart-base/"
    # gentype="vanilla"

    # ## 3. dailydialog MERGE ours
    # task="daily_dialog"
    # subset=None
    # model_path="./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/manystep500001000104118e-50.010.60.70.75finally/"
    # gentype="ER"

    # ## 4. dailydialog MERGE-onlyER ours
    # task="daily_dialog"
    # subset=None
    # model_path="./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/onlyER150001000104008e-50.010.60.70.75finally/"
    # gentype="ER"

    # ## 5. commonGen mpcformer
    # task="common_gen"
    # subset=None
    # model_path="./stage1_ckpts/common_gen-epoch3-lr5e-05-bs32gpt2/mpcformer1010004108e-50.010.00.00.0finally/"
    # gentype="vanilla"

    # ## 6. daily dialog mpcformer
    # task="daily_dialog"
    # subset=None
    # model_path="./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/mpcformer1010004108e-50.010.00.00.0finally/"
    # gentype="vanilla"

    # ## 7. multiwoz gpt2 mpcformer
    # task="multiwoz_nlg"
    # subset=None
    # model_path="./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/mpcformer1010004108e-50.010.00.00.0finally/"
    # gentype="vanilla"

    # ## 8. commongen gpt2 MERGE
    # task="common_gen"
    # subset=None
    # model_path="./stage1_ckpts/common_gen-epoch3-lr5e-05-bs32gpt2/longStep500001000104113e-40.010.60.70.75/"
    # gentype="ER"

    # ## 9. commongen gpt2 MERGE only ER
    # task="common_gen"
    # subset=None
    # model_path="./stage1_ckpts/common_gen-epoch3-lr5e-05-bs32gpt2/onlyER150001000104008e-50.010.60.70.75finally/"
    # gentype="ER"

    # ## 10. commongen gpt2 MERGE only MM
    # task="common_gen"
    # subset=None
    # model_path="./stage1_ckpts/common_gen-epoch3-lr5e-05-bs32gpt2/longStep500001000104023e-40.010.60.70.75finally/"
    # gentype="vanilla"

    # ## 11. dailydialog gpt2 MERGE only MM
    # task="daily_dialog"
    # subset=None
    # # model_path="./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/onlyMM150001000104028e-50.010.60.70.75finally/"
    # model_path="./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/onlyMM150001000104028e-50.010.60.70.75/"
    # gentype="vanilla"

    # ## 12. multiwoz t5 mpcformer
    # task="multiwoz_nlg"
    # subset=None
    # model_path="./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4t5-small/mpcformer1010004108e-50.010.00.00.0finally/"
    # gentype="vanilla"

    # ## 13. multiwoz bart mpcformer
    # task="multiwoz_nlg"
    # subset=None
    # model_path="./stage1_ckpts/multiwoz_nlg-epoch6-lr5e-5-bs32bart-base/mpcformer1010004108e-50.010.00.00.0finally/"
    # gentype="vanilla"

    # ## 14. multiwoz gpt2 onlyMM
    # task="multiwoz_nlg"
    # subset=None
    # model_path="./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/onlyMM1000104028e-40.010.60.70.75finally/"
    # gentype="vanilla"

    # ## 15. multiwoz gpt2 onlyER
    # task="multiwoz_nlg"
    # subset=None
    # model_path="./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/onlyMM1000104008e-40.010.60.70.75finally/"
    # gentype="ER"

    # ## 16. multiwoz t5 MERGE
    # task="multiwoz_nlg"
    # subset=None
    # model_path="./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4t5-small/addQuad1000104119e-50.010.60.70.75finally/"
    # gentype="ER"

    # ## 17. multiwoz bart MERGE
    # task="multiwoz_nlg"
    # subset=None
    # model_path="./stage1_ckpts/multiwoz_nlg-epoch6-lr5e-5-bs32bart-base/addQuad1000104013e-40.010.60.70.75finally/"
    # gentype="ER"

    # ## 15. multiwoz gpt2 MERGE
    # task="multiwoz_nlg"
    # subset=None
    # model_path="./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/longstep1004118e-40.01/"
    # gentype="ER"

    # ## 16. dailydialog gpt2 MERGE
    # task="daily_dialog"
    # subset=None
    # # model_path="./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/___withConstantMatrix/noSoftmax500001000104118e-50.010.60.70.75finally/"
    # model_path="./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/noSoftmax500001000104018e-50.010.60.70.75finally/"
    # # model_path="./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/ActivationnoSoftmax500001000104018e-50.010.60.70.75epoch20/"
    # gentype="vanilla"
    # # gentype="ER"

    # ## 16. dailydialog gpt2 MERGE
    # task="multiwoz_nlg"
    # subset=None
    # # model_path="./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/___withConstantMatrix/noSoftmax500001000104118e-50.010.60.70.75finally/"
    # model_path="./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4t5-small/index_after_softmax1000104119e-50.010.60.70.75epoch0/"
    # # model_path="./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/ActivationnoSoftmax500001000104018e-50.010.60.70.75epoch20/"
    # gentype="vanilla"
    # # gentype="ER"

    ## 17. dailydialog gpt2 MERGE
    task="multiwoz_nlg"
    subset=None
    model_path="./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/indexAfterSoftMax500001000104018e-50.010.60.70.75/"
    gentype="vanilla"
    # gentype="ER"


    if "bart" in model_path or "t5" in model_path:
        withsep=False
    
    # cuda_num=1
    cuda_num=0

    # gentype="ER"

    from inference_cp import Inference

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

        # # # # print(seqls[0])
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


## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


