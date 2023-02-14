#!/bin/bash
######################################################################
#0.2.DISTILL_LAYERNORM ---

# Further distill based on the first step KD.

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created: 14 二月 2023
######################################################################

######################### Commentary ##################################
##  
######################################################################

export python=/home/liangzi/anaconda3/envs/HE/bin/python3
export root_dir="/home/liangzi/mpcGen/nlg/"

export epochs=3
export lr=3e-5
export device="7"
# export device="cpu"
export batch_size=1
export task="web_nlg"
export max_seq_length=128

export teach_ckpt="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/___withConstantMatrixDistilled111410/"
export stu_ckpt=${teach_ckpt}

export using_entropy=1
export using_softLabel=1
export tau=4
export using_interKL=1
export using_wordEmbedMSE=1
export using_quadacti=1

export using_simLN=1
export weight_decay=0.5

# export using_wordEmbedMSE=0
export stu_save_ckpt=${stu_ckpt}LayerNorm${using_entropy}${using_softLabel}${using_interKL}${tau}${using_quadacti}${using_simLN}${weight_decay}

export lonelyLongOverallPath="./distillModelResTest.log"

export board_name=$stu_save_ckpt

# export models=("bert-base-uncased")
# export devices=(1 2 3 4 5 6 7 0)
# export devices=(7 2 3 4 5 6 7 0)
# export tasks=(cola mnli_matched mnli_mismatched mrpc qnli qqp\
# 		 rte wnli)
# export tasks=(cola)
# export time_list=(1)

${python} trains3.py \
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
	--using_quadacti=${using_quadacti}\
	--using_simLN=${using_simLN}\
	--board_name=${board_name}\
	--weight_decay=${weight_decay}\
	--root_dir=$root_dir


echo "RUNNING 0.2.distill_layerNorm.sh DONE."
# 0.2.distill_layerNorm.sh ends here
