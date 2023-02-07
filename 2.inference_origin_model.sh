#!/bin/bash
######################################################################
#2.INFERENCE_ORIGIN_MODEL ---

# Only For Inference

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2022, ZiLiang, all rights reserved.
# Created: 26 十二月 2022
######################################################################

export python=/home/liangzi/anaconda3/envs/HE/bin/python3
export root_dir="/home/liangzi/he_transformer/newglue/"

##--------------------------------------------------------------------
export device="6"
export epochs=30
export batch_size=32
export lr=3e-5
export max_seq_length=128
export pretrained_model_path="bert-tiny" 
export lonelyLongOverallPath="./20221223-GLUEAll.log"
# export models=("bert-tiny" "bert-base-uncased")
export models=("bert-base-uncased")
# export models=("bert-tiny" "bert-base-uncased")
# export devices=(1 2 3 4 5 6 7 0)
export devices=(6 6 6 6 6 6 6 6)
export tasks=(cola mnli_matched mnli_mismatched mrpc qnli qqp\
		 rte wnli)
# export tasks=(cola mnli_matched mnli_mismatched mrpc qnli qqp\
		 # rte sst2 wnli)
# export tasks=(ax cola mnli_matched mnli_mismatched mrpc qnli qqp\
		 # rte sst2 stsb wnli)
export time_list=(1)

for i in `seq 1 ${#tasks[@]}`;
# for langname in ${langs[*]};
do
    for model in ${models[*]};
    do
    export task=${tasks[$i-1]}
    export device=${devices[$i-1]}
    echo "===================================================="
    echo "model_name: $modelname; device: $device; task: $task"
    echo "===================================================="

    ${python} train.py \
	    --train=0 \
	    --epochs=${epochs} \
	    --lr=${lr} \
	    --cuda_num=${device} \
	    --batch_size=${batch_size} \
	    --task=${task} \
	    --max_seq_length=${max_seq_length} \
	    --pretrained_model_path=${model} \
	    --root_dir=$root_dir
    done
done


echo "RUNNING 2.inference_origin_model.sh DONE."
# 2.inference_origin_model.sh ends here
