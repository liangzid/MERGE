#!/bin/bash
######################################################################
#0.1.DISTILL_TRAIN ---

# RUNNING NLG DISTILLATION TRAINING FOR GPT2

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created: 10 二月 2023
######################################################################

export python=/home/liangzi/anaconda3/envs/HE/bin/python3
export root_dir="/home/liangzi/mpcGen/nlu/"

export epochs=30
export lr=3e-5
export device="6"
export batch_size=32
export task="GEM/web_nlg"
export max_seq_length=128

# export teach_ckpt=${root_dir}/save_models/saved_bert-base-uncased_taskcola-epoch30-lr3e-05-bs32
export teach_ckpt="/home/liangzi/models/gpt2/"
export stu_ckpt=${teach_ckpt}___withConstantMatrix

export using_entropy=1
export using_softLabel=1
export tau=1
export using_interKL=1
export using_wordEmbedMSE=1
export stu_save_ckpt=${stu_ckpt}Distilled${using_entropy}${using_softLabel}${using_interKL}${tau}

export lonelyLongOverallPath="./distillModelResTest.log"

# export models=("bert-base-uncased")
# export devices=(1 2 3 4 5 6 7 0)
# export devices=(7 2 3 4 5 6 7 0)
# export tasks=(cola mnli_matched mnli_mismatched mrpc qnli qqp\
# 		 rte wnli)
# export tasks=(cola)
# export time_list=(1)

${python} trains2.py \
	--train=1 \
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
	--using_wordEmbedMSE=${using_wordEmbedMSE}\
	--tau=${tau}\
	--root_dir=$root_dir


echo "RUNNING 0.1.distill_train.sh DONE."
# 0.1.distill_train.sh ends here
