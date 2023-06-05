#!/bin/bash
######################################################################
#1.33.BART_NOSOFTMAX_TRAIN --- 

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created: 22 五月 2023
######################################################################

export python=/home/liangzi/anaconda3/envs/HE/bin/python3
export root_dir="/home/liangzi/mpcgen/nlg/"

export epochs=3000
export step=100000
# export lr=8e-4
export lr=8e-5
export max_seq_length=128

export batch_size=32
# export batch_size=1
export task="multiwoz_nlg"
# export teach_ckpt="./stage1_ckpts/multiwoz_nlg-epoch6-lr5e-5-bs32bart-base"
export teach_ckpt="./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4t5-small/"
export device="1"

export stu_ckpt=${teach_ckpt}

export using_entropy=1
export using_softLabel=0
export tau=4
export using_interKL=0
export using_wordEmbedMSE=0
export using_COSEm=1
export using_NEGAEm=0

##############################################################

# ## only ER
# export using_quadacti=0 ##### now add the quadtic option.
# export using_simLN=0
# export no_res=0
# export no_softmax=1
# export stu_save_ckpt=${stu_ckpt}onlyER${step}${using_entropy}${using_softLabel}${using_interKL}${using_wordEmbedMSE}${using_COSEm}${using_NEGAEm}${tau}${using_quadacti}${using_simLN}${lr}${weight_decay}${dropout_rate}${noise}${lamda}

## MERGE
export using_quadacti=1 ##### now add the quadtic option.
export using_simLN=1
export no_res=0
export no_softmax=1

export weight_decay=0.01
export dropout_rate=0.75
export noise=0.95
# export noise=0.2
export lamda=0.9

export stu_save_ckpt=${stu_ckpt}BigNoise${step}${using_entropy}${using_softLabel}${using_interKL}${using_wordEmbedMSE}${using_COSEm}${using_NEGAEm}${tau}${using_quadacti}${using_simLN}${lr}${weight_decay}${dropout_rate}${noise}${lamda}

##############################################################

export lonelyLongOverallPath="./distillModelResTest.log"

export board_name=$stu_save_ckpt

${python} train_slide.py \
	--train=1 \
	--no_softmax=1 \
	--epochs=${epochs} \
	--train_step=${step} \
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
	--using_COSEm=${using_COSEm}\
	--using_NEGAEm=${using_NEGAEm}\
	--tau=${tau}\
	--using_quadacti=${using_quadacti}\
	--using_simLN=${using_simLN}\
	--board_name=${board_name}\
	--weight_decay=${weight_decay}\
	--dropout_rate=${dropout_rate}\
	--dropout_rate=${noise}\
	--lamda=${lamda}\
	--root_dir=$root_dir

echo "RUNNING 1.33.bart_nosoftmax_train.sh DONE."
# 1.33.bart_nosoftmax_train.sh ends here
