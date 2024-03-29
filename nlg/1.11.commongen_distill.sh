#!/bin/bash
######################################################################
#1.11.COMMONGEN_DISTILL ---

# distill common gen.

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created: 27 三月 2023
######################################################################

export python=/home/liangzi/anaconda3/envs/HE/bin/python3
export root_dir="/home/liangzi/mpcGen/nlg/"

export epochs=1000
export step=50000
# export lr=3e-5
# export lr=8e-5
export lr=8e-4
# export device="cpu"
export batch_size=64
# export task="web_nlg"
# export task="e2e_nlg"
# export task="daily_dialog"
# export task="multiwoz_nlg"
export task="common_gen"
export max_seq_length=128

export teach_ckpt="./stage1_ckpts/common_gen-epoch3-lr5e-05-bs32gpt2/"

export stu_ckpt=${teach_ckpt}

export using_entropy=1
export using_softLabel=0
export tau=4
export using_interKL=0
export using_wordEmbedMSE=0
export using_COSEm=1
export using_NEGAEm=0

##############################################################

# ## method 3
# export using_quadacti=0 ##### now add the quadtic option.
# export using_simLN=0
# export lamda=0.75
# export device="0"

# ## method 6
# export using_quadacti=1 ##### now add the quadtic option.
# export using_simLN=1
# export lamda=0.5
# export device="0"

## method 7
export no_res=1
export using_quadacti=0 ##### now add the quadtic option.
export using_simLN=1 # 2 denotes do not use simLN, but use the approximation model
export lamda=0.75
export device="1"

##############################################################

export weight_decay=0.01
export dropout_rate=0.6
export noise=0.7

# export using_wordEmbedMSE=0
export stu_save_ckpt=${stu_ckpt}nores${no_res}${step}${using_entropy}${using_softLabel}${using_interKL}${using_wordEmbedMSE}${using_COSEm}${using_NEGAEm}${tau}${using_quadacti}${using_simLN}${lr}${weight_decay}${dropout_rate}${noise}${lamda}

export lonelyLongOverallPath="./distillModelResTest.log"

export board_name=$stu_save_ckpt

${python} train_slide.py \
	--train=1 \
	--epochs=${epochs} \
	--train_step=${step} \
	--no_res=${no_res} \
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

echo "RUNNING 1.11.commongen_distill.sh DONE."
# 1.11.commongen_distill.sh ends here
