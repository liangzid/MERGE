from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from transformers import AutoModelForSequenceClassification 
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score

from transformers import glue_compute_metrics,glue_output_modes,glue_tasks_num_labels
glue_output_modes["sst2"]="classification"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

import json
import numpy as np
import argparse

# # you can use any of the following config names as a second argument:
# "ax", "cola", "mnli", "mnli_matched", 
# "mnli_mismatched", "mrpc", "qnli", "qqp", 
# "rte", "sst2", "stsb", "wnli"

glue_dataset_ls=["ax","cola","mnli","mnli_matched","mnli_mismatched",
                 "mrpc","qnli","qqp","rte","sst2","stsb","wnli"]

def convert_text_to_ids_segment(text,text1, max_sentence_length,tokenizer):

    if text1 is None:
        res= tokenizer(text,padding="max_length", truncation=True,
                                        max_length=max_sentence_length,
                                        return_tensors="pt")
        ids,segs,masks=res.input_ids,res.token_type_ids,res.attention_mask
    else:
        res = tokenizer(text,text1,truncation=True,
                                        padding="max_length",
                                        max_length=max_sentence_length,
                                        return_tensors="pt")
        ids,segs,masks=res.input_ids,res.token_type_ids,res.attention_mask
    return ids,segs,masks


## set dataset
def getDataset(tokenizer,labelmap,msl=128,glue_task="cola",mode="train"):
    dataset=load_dataset("glue",glue_task)[mode]

    sep_token=tokenizer.sep_token
    num_samples=len(dataset)

    inps=torch.zeros((num_samples,msl),dtype=torch.long)
    attentions=torch.zeros((num_samples,msl),dtype=torch.long)
    typeids=torch.zeros((num_samples,msl),dtype=torch.long)
    labels=torch.zeros((num_samples,1),dtype=torch.long)

    item=dataset[0]

    # single sentence formation
    if "sentence" in item and "question" not in item:

        inpt=[item["sentence"] for item in dataset]
        indexs, segs, atts = convert_text_to_ids_segment(inpt,None,
                                                max_sentence_length=msl,
                                                tokenizer=tokenizer)
        labels=[item["label"] for item in dataset]
    # double sentence formation
    elif "premise" in item:
        t1=[x["premise"] for x in dataset]
        t2=[x["hypothesis"] for x in dataset]
        indexs, segs, atts = convert_text_to_ids_segment(t1,t2,
                                                max_sentence_length=msl,
                                                tokenizer=tokenizer)

        labels=[item["label"] for item in dataset]
    elif "sentence1" in item:
        t1=[x["sentence1"] for x in dataset]
        t2=[x["sentence2"] for x in dataset]
        indexs, segs, atts = convert_text_to_ids_segment(t1,t2,
                                                max_sentence_length=msl,
                                                tokenizer=tokenizer)
        labels=[item["label"] for item in dataset]
    elif "question" in item:
        t1=[x["question"] for x in dataset]
        t2=[x["sentence"] for x in dataset]
        inpt=item["question"]+sep_token+item["sentence"]
        indexs, segs, atts = convert_text_to_ids_segment(t1,t2,
                                                max_sentence_length=msl,
                                                tokenizer=tokenizer)

        labels=[item["label"] for item in dataset]
    elif "question1" in item:
        t1=[x["question1"] for x in dataset]
        t2=[x["question2"] for x in dataset]
        indexs, segs, atts = convert_text_to_ids_segment(t1,t2,
                                                max_sentence_length=msl,
                                                tokenizer=tokenizer)
        labels=[item["label"] for item in dataset]
    else:
        print("Error, unseen dataset formation.")
        assert 1==0

    labels=torch.tensor(labels,dtype=torch.long)
    dataset = TensorDataset(indexs,atts,segs,labels)
    return dataset


def train(model, optimizer, train_loader,val_loader,
          task,path,
          EPOCH,LR,DEVICE,
          batch_size=32,
          ):

    past_acc=-1.
    for epoch in range(EPOCH):
        correct = 0
        undetected = 0
        detected = 0

        print(f"-------EPOCH {epoch}-------------")
        for i,(inputs,attentions,typeids,labels) in enumerate(train_loader):
            inputs,attentions,typeids,labels=inputs.to(DEVICE),\
                attentions.to(DEVICE),typeids.to(DEVICE),labels.to(DEVICE)

            outputs = model(inputs,attentions,
                            typeids,labels=labels)

            prediction = torch.nn.functional.softmax(outputs.logits,dim=1)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predict_result = torch.nn.functional.softmax(outputs.logits, dim=1)
            predict_result = torch.argmax(predict_result, dim=1).cpu().numpy()

            if i%300==0:
                print(f"loss:{loss.item()}")
            
        acc=test(test_loader=val_loader,model=model,task=task,
                batch_size=batch_size,DEVICE=DEVICE)
        if acc>past_acc:
            model.save_pretrained(path)
            past_acc=acc

def test(test_loader,model,task,batch_size=32,DEVICE="cpu"):
    correct = 0
    predict_list=[]
    ground_truth_list=[]

    glue_output_mode=glue_output_modes[task]

    print("--------TEST---------")
    for i,(inputs,attentions,type_ids,labels) in enumerate(test_loader):
        inputs,attentions,type_ids,labels=inputs.to(DEVICE),\
            attentions.to(DEVICE),\
            type_ids.to(DEVICE),labels.to(DEVICE)
        outputs = model(inputs,attentions,type_ids,labels=labels)
        loss = outputs.loss

        predict_result = torch.nn.functional.softmax(outputs.logits, dim=1)
        if glue_output_mode=="classification":
            predict_result = torch.argmax(predict_result, dim=1).cpu().numpy()
        else:
            pass
            # predict_result = torch.argmax(predict_result, dim=1).cpu().numpy()
        predict_list.extend(predict_result)
        
        for batch in labels:
            ground_truth_list.append(int(batch))

    ## calculate precision recall and f1 score
    ylbl=np.array(ground_truth_list)
    ypl=np.array(predict_list)

    pre=precision_score(ylbl,ypl)
    rec=recall_score(ylbl,ypl)
    f1=f1_score(ylbl,ypl)
    acc=accuracy_score(ylbl,ypl)

    # print(f"precision: {pre}")
    # print(f"recall: {rec}")
    print(f"f1 score: {f1}")
    print(f"accuracy: {acc}")
    print("=================================")

    # print(ylbl)
    # print(ypl)
    scores=glue_compute_metrics(task,ypl,ylbl)
    print(scores)
    return acc


def preprocess(tokenizer,device,batch_size,task):

    origin2new_map={}
    if task in ["mnli_mismatched", "mnli_matched"]:
        trainset=getDataset(tokenizer,origin2new_map,glue_task="mnli",mode="train")
    else:
        trainset=getDataset(tokenizer,origin2new_map,glue_task=task,mode="train")
    valset=getDataset(tokenizer,origin2new_map,glue_task=task,mode="validation")
    testset=getDataset(tokenizer,origin2new_map,glue_task=task,mode="test")

    train_loader=DataLoader(trainset,batch_size=batch_size,
                            shuffle=True,drop_last=False)
    val_loader=DataLoader(valset,batch_size=batch_size,
                            shuffle=False,drop_last=False)
    test_loader=DataLoader(testset,batch_size=batch_size,
                            shuffle=False,drop_last=False)
    return train_loader,val_loader,test_loader

def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int,
                        required=False, help='训练的轮次')
    parser.add_argument('--lr', default=3e-4, type=float,
                        required=False, help='学习率')
    parser.add_argument('--cuda_num', default='6', type=str, required=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        required=False, help='训练batch size')
    parser.add_argument('--task', default="cola", type=str,
                        required=True,)
    parser.add_argument("--max_seq_length", default=128,
                        type=int, required=False, help="模型的最大输入长度")
    parser.add_argument("--train", default=1, type=int,
                        required=True, help="用以决定是训练模式还是测试模式")
    parser.add_argument('--pretrained_model_path', default='bert-tiny',
                        type=str, required=True,)
    parser.add_argument('--root_dir', default='/home/liangzi/he_transformer/newglue/',
                        type=str, required=False,)

    return parser.parse_args()

def main2():
    glue_dataset_ls=["ax","cola","mnli","mnli_matched","mnli_mismatched",
                 "mrpc","qnli","qqp","rte","sst2","stsb","wnli"]
    args=setup_train_args()
    EPOCH = args.epochs
    LR = args.lr
    if args.cuda_num=="cpu":
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(f"cuda:{args.cuda_num}")
    BATCH_SIZE =args.batch_size
    task=args.task
    PATH = f'{args.root_dir}/save_models/saved_{args.pretrained_model_path}_task{task}-epoch{EPOCH}-lr{LR}-bs{BATCH_SIZE}'

    prefix_path="/home/liangzi/models/"
    model_name=args.pretrained_model_path
    if "/home" in model_name:
        frmpth=model_name
    else:
        frmpth=prefix_path+model_name
    
    model = AutoModelForSequenceClassification.from_pretrained(frmpth)
    tokenizer = AutoTokenizer.from_pretrained(frmpth)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model = model.to(DEVICE)

    train_loader,val_loader,\
        test_loader = preprocess(tokenizer=tokenizer,device=DEVICE,
                                batch_size=BATCH_SIZE,task=task)

    if args.train==1:
        #============================================
        train(model=model, optimizer=optimizer,
              train_loader=train_loader,val_loader=val_loader,
              task=task,path=PATH,
              batch_size=BATCH_SIZE,
              EPOCH=EPOCH,LR=LR,
              DEVICE=DEVICE,)
        # model.save_pretrained(PATH)
        tokenizer.save_pretrained(PATH)
        #============================================

    model=model.from_pretrained(PATH)
    model.to(DEVICE)
    model.eval()
    
    test(test_loader=val_loader,model=model,task=task,
         batch_size=BATCH_SIZE,DEVICE=DEVICE)

    # test(test_loader=test_loader,model=model,task=task,
    #      batch_size=BATCH_SIZE,DEVICE=DEVICE)
    

def main1():
    EPOCH = 40
    # LR = 5e-5 
    LR = 5e-5 
    # DEVICE = torch.device("cuda:1")
    DEVICE = torch.device("cpu")
    BATCH_SIZE =32
    glue_dataset_ls=["ax","cola","mnli","mnli_matched","mnli_mismatched",
                 "mrpc","qnli","qqp","rte","sst2","stsb","wnli"]
    task="cola"
    PATH = f'saved_task{task}-epoch{EPOCH}-lr{LR}-bs{BATCH_SIZE}'

    prefix_path="/home/liangzi/models/"
    model_name="bert-tiny"
    frmpth=prefix_path+model_name
    
    model = AutoModelForSequenceClassification.from_pretrained(frmpth)
    tokenizer = AutoTokenizer.from_pretrained(frmpth)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model = model.to(DEVICE)

    train_loader,val_loader,\
        test_loader = preprocess(tokenizer=tokenizer,device=DEVICE,
                                batch_size=BATCH_SIZE,task=task)

    # #============================================
    # train(model=model, optimizer=optimizer,
    #       train_loader=train_loader, batch_size=BATCH_SIZE,
    #       EPOCH=EPOCH,LR=LR,
    #       DEVICE=DEVICE,)
    # model.save_pretrained(PATH)
    # tokenizer.save_pretrained(PATH)
    # #============================================

    model=model.from_pretrained(PATH)
    model.to(DEVICE)
    model.eval()
    
    test(test_loader=val_loader,model=model,task=task,
         batch_size=BATCH_SIZE,DEVICE=DEVICE)

    # test(test_loader=test_loader,model=model,task=task,
    #      batch_size=BATCH_SIZE,DEVICE=DEVICE)
    

if __name__=="__main__":
    # main1()
    main2()
