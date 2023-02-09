#!/bin/bash
######################################################################
#3.DISTILL_TRAIN ---

# Training Model with Knowledge Distillation.

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2022, ZiLiang, all rights reserved.
# Created: 29 十二月 2022
######################################################################

######################### Commentary ##################################
##  
######################################################################

export python=/home/liangzi/anaconda3/envs/HE/bin/python3
export root_dir="/home/liangzi/he_transformer/newglue/"

##--------------------------------------------------------------------
export device="6"
export epochs=30
export batch_size=32
export lr=3e-5
export max_seq_length=128

export teach_ckpt=${root_dir}/save_models/saved_bert-base-uncased_taskcola-epoch30-lr3e-05-bs32
export stu_ckpt=${teach_ckpt}___withConstantMatrix
export using_entropy=1
export using_softLabel=1
export using_interKL=1
export tau=1
export stu_save_ckpt=${stu_ckpt}___Distilled_${using_entropy}${using_softLabel}${using_interKL}${tau}


export lonelyLongOverallPath="./1229-GLUEAll.log"
# export models=("bert-tiny" "bert-base-uncased")
export models=("bert-base-uncased")
export devices=(1 2 3 4 5 6 7 0)
export devices=(7 2 3 4 5 6 7 0)
export tasks=(cola mnli_matched mnli_mismatched mrpc qnli qqp\
		 rte wnli)
export tasks=(cola)
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
    echo "teacher ckpt: $teach_ckpt;"
    echo "student ckpt: $stu_ckpt;"
    echo "student save ckpt: $stu_save_ckpt;"
    echo "temperature: $tau;"
    echo "===================================================="

    ${python} distill_train.py \
	    --train=0 \
	    --epochs=${epochs} \
	    --lr=${lr} \
	    --cuda_num=${device} \
	    --batch_size=${batch_size} \
	    --task=${task} \
	    --max_seq_length=${max_seq_length} \
	    --teach_ckpt=${teach_ckpt}\
	    --stu_ckpt=${stu_ckpt}\
	    --stu_save_ckpt=${stu_save_ckpt}\
	    --using_entropy=${using_entropy}\
	    --using_softLabel=${using_softLabel}\
	    --using_interKL=${using_interKL}\
	    --tau=${tau}\
	    --root_dir=$root_dir

    done
done


echo "RUNNING 3.distill_train.sh DONE."
# 3.distill_train.sh ends here
