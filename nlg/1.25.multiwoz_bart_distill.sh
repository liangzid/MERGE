#!/bin/bash
######################################################################
#1.25.MULTIWOZ_BART_DISTILL --- 

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created: 14 四月 2023
######################################################################

######################### Commentary ##################################
##  
######################################################################

export python=/home/liangzi/anaconda3/envs/HE/bin/python3
export root_dir="/home/liangzi/mpcGen/nlg/"

export epochs=10000
export step=50000
# export lr=3e-5
export lr=9e-5
# export lr=3e-4
# export device="cpu"
export batch_size=16
# export task="web_nlg"
# export task="e2e_nlg"
# export task="daily_dialog"
export task="multiwoz_nlg"
export max_seq_length=128

# export teach_ckpt="./stage1_ckpts/e2e_nlg-epoch3-lr5e-05-bs1gpt2/"
# export teach_ckpt="./stage1_ckpts/daily_dialog-epoch3-lr5e-05-bs4gpt2/"

export teach_ckpt="./stage1_ckpts/multiwoz_nlg-epoch6-lr5e-5-bs32bart-base/"

# export teach_ckpt="./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4gpt2/"
# export teach_ckpt="./stage1_ckpts/multiwoz_nlg-epoch3-lr5e-05-bs4t5-small/"
# export teach_ckpt="./stage1_ckpts/web_nlg-epoch3-lr5e-05-bs1gpt2/"

export stu_ckpt=${teach_ckpt}
# export stu_ckpt=${teach_ckpt}"___withConstantMatrix"

export using_entropy=1
export using_softLabel=0
export tau=4
export using_interKL=0
export using_wordEmbedMSE=0
export using_COSEm=1
export using_NEGAEm=0

########################################################

# ## method 3
# export using_quadacti=0 ##### now add the quadtic option.
# export using_simLN=0
# export lamda=0.75
# export device="5"
# export weight_decay=0.01
# export dropout_rate=0.6
# export noise=0.7

# ## only LM 结论：不收敛
# export using_quadacti=0 ##### now add the quadtic option.
# export using_simLN=1
# export lamda=0.5
# export device="0"
# export weight_decay=0.01
# export dropout_rate=0.1
# export noise=0.5
# export using_COSEm=1 # do not need for ER

## overall
export using_quadacti=0 ##### now add the quadtic option.
export using_simLN=1
export lamda=0.75
export device="3"
export weight_decay=0.01
export dropout_rate=0.6
export noise=0.7

########################################################


# export lamda=0.5

# export using_wordEmbedMSE=0
export stu_save_ckpt=${stu_ckpt}addQuad${using_entropy}${using_softLabel}${using_interKL}${using_wordEmbedMSE}${using_COSEm}${using_NEGAEm}${tau}${using_quadacti}${using_simLN}${lr}${weight_decay}${dropout_rate}${noise}${lamda}

export lonelyLongOverallPath="./distillModelResTest.log"

export board_name=$stu_save_ckpt

${python} train_slide.py \
	--train=1 \
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


echo "RUNNING 1.25.multiwoz_bart_distill.sh DONE."
# 1.25.multiwoz_bart_distill.sh ends here
