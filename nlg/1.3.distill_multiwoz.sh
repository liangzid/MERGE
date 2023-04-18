#!/bin/bash
######################################################################
#1.3.DISTILL_MULTIWOZ --- 

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created: 25 二月 2023
######################################################################

export python=/home/liangzi/anaconda3/envs/HE/bin/python3
export root_dir="/home/liangzi/mpcGen/nlg/"

export epochs=1000
export step=50000
# export lr=3e-5
export lr=8e-4
# export lr=3e-4
export batch_size=16
# export task="web_nlg"
# export task="e2e_nlg"
export task="multiwoz_nlg"
export max_seq_length=128

export teach_ckpt="./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/"
# export teach_ckpt="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropoutfinally/"
export stu_ckpt=${teach_ckpt}

export using_entropy=1
export using_softLabel=0
export tau=4
export using_interKL=0
export using_wordEmbedMSE=0
export using_quadacti=0
export using_COSEm=1
export using_NEGAEM=0

export using_quadacti=1 ##### now add the quadtic option.
export using_simLN=1
export lamda=0.75
export device="3"
export weight_decay=0.01
export dropout_rate=0.6
export noise=0.7

# export using_wordEmbedMSE=0
export stu_save_ckpt=${stu_ckpt}longstep${using_entropy}${using_softLabel}${using_interKL}${tau}${using_quadacti}${using_simLN}${lr}${weight_decay}

export lonelyLongOverallPath="./distillModelResTest.log"

export board_name=$stu_save_ckpt

${python} train_slide.py \
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


echo "RUNNING 1.3.distill_multiwoz.sh DONE."
# 1.3.distill_multiwoz.sh ends here
