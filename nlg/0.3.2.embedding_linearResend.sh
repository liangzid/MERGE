#!/bin/bash
######################################################################
#0.3.2.EMBEDDING_LINEARRESEND ---

# test embedding resend.

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created: 20 二月 2023
######################################################################

export python=/home/liangzi/anaconda3/envs/HE/bin/python3
export root_dir="/home/liangzi/mpcGen/nlg/"

export epochs=10
# export lr=3e-5
export lr=8e-5
# export lr=3e-4
export device="5"
export batch_size=1
export task="web_nlg"
export max_seq_length=128

export teach_ckpt="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/WithEmbedResendDistilled1114008e-50.01trainmodel/ReTraining1114008e-50.01_testdropouttrainmodel/"
export stu_ckpt=${teach_ckpt}

export using_entropy=1
export using_softLabel=1
export tau=4
export using_interKL=1
export using_wordEmbedMSE=1
export using_quadacti=0

export using_simLN=0
export weight_decay=0.01

# export using_wordEmbedMSE=0
export stu_save_ckpt=${stu_ckpt}WithLinearProjection

export lonelyLongOverallPath="./distillModelResTest.log"

export board_name=$stu_save_ckpt

${python} train_slide.py \
	--train=1 \
	--using_prolayer=1 \
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

echo "RUNNING 0.3.2.embedding_linearResend.sh DONE."
# 0.3.2.embedding_linearResend.sh ends here
