"""
======================================================================
FOR_DATASET2 --- 

New for dataset.

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


def getFinetunedSet2(tokenizer,
                    max_sentence_length=256,
                    task="web_nlg",subset="release_v2",
                    only_decoder=True):
    """
    For Downstream Tasks based on Conditional Generation.
    task and subtask enums:
    + GEM/web_nlg
        + en
        + ru
    + e2e_nlg, subset:none
    """
    sep_token="<|sep|>"
    sep_token_id=tokenizer.encode([sep_token])[0]
    print(f"sep token id: {sep_token_id}")
    # sep_token=tokenizer.sep_token
    eos_token=tokenizer.eos_token

    def getSet(split="train"):
        # print(subset)
        if subset is not None:
            train_set=load_dataset(task,subset,split=split)
        else:
            train_set=load_dataset(task,split=split)
        # print(train_set)

        inps=[]
        outs=[]
        if "web_nlg" in task:
            for x in train_set:
                inps.append(" ; ".join(x["modified_triple_sets"]\
                                       ["mtriple_set"][0]))
                outs.append(x["lex"]["text"][0])

        elif "e2e_nlg" in task:
            for x in train_set:
                inps.append(x["meaning_representation"])
                outs.append(x["human_reference"])
        
        if only_decoder:
            outs=[inps[i]+sep_token+outs[i]+eos_token\
                for i in range(len(train_set))]

            outss=tokenizer(outs,padding="longest",
                            truncation=True,
                        max_length=max_sentence_length,return_tensors="pt")
            inps=outss.input_ids
            for x in inps:
                sep_idx=x.index(sep_token_id)
            labels

            
            dset=TensorDataset(outss.input_ids,outss.attention_mask,
                            )





        else:
            inps=tokenizer(inps,padding="longest",truncation=True,
                           max_length=max_sentence_length,
                           return_tensors="pt")
            outs=[x+eos_token for x in outs]
            outs=tokenizer(outs,padding="longest",truncation=True,
                           max_length=max_sentence_length,
                           return_tensors="pt")
            dset=TensorDataset(inps.input_ids,
                               inps.attention_mask,
                               outs.input_ids,
                            )
        return dset

    if "web_nlg" in task:
        names=["train","dev","test"]
    elif "e2e_nlg" in task:
        names=["train","validation","test"]
    return getSet(names[0]),getSet(names[1]),getSet(names[2])































## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


