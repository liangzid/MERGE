#!/bin/bash
######################################################################
#1.4.VARY_LOSS_E2E --- 

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created: 28 二月 2023
######################################################################

export python=/home/liangzi/anaconda3/envs/HE/bin/python3
export root_dir="/home/liangzi/mpcGen/nlg/"

export epochs=5
# export lr=3e-5
export lr=8e-5
# export lr=3e-4
export device="5"
export batch_size=4
# export task="web_nlg"
export task="e2e_nlg"
export max_seq_length=128

export teach_ckpt="./stage1_ckpts/e2e_nlg-epoch3-lr5e-05-bs1gpt2/"
export stu_ckpt=${teach_ckpt}

export using_entropy=1
export using_softLabel=1
export tau=4
export using_interKL=1
export using_wordEmbedMSE=1
export using_COSEm=1
export using_NEGAEm=1
export using_quadacti=0

export using_simLN=0
export weight_decay=0.01
export dropout_rate=0.7
export noise=0.7

# export using_wordEmbedMSE=0
export stu_save_ckpt=${stu_ckpt}VaryLoss${using_entropy}${using_softLabel}${using_interKL}${tau}${using_wordEmbedMSE}${using_COSEm}${using_NEGAEm}${using_quadacti}${using_simLN}${lr}${weight_decay}${dropout_rate}${noise}

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
	--using_COSEm=${using_COSEm}\
	--using_NEGAEm=${using_NEGAEm}\
	--tau=${tau}\
	--using_quadacti=${using_quadacti}\
	--using_simLN=${using_simLN}\
	--board_name=${board_name}\
	--weight_decay=${weight_decay}\
	--dropout_rate=${dropout_rate}\
	--dropout_rate=${noise}\
	--root_dir=$root_dir

echo "RUNNING 1.4.vary_loss_e2e.sh DONE."
# 1.4.vary_loss_e2e.sh ends here
