#!/bin/bash
######################################################################
#4.ROBUSTNESS_ANANLYSIS ---

# analysis the robustness of input word embeddings.

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created:  8 五月 2023
######################################################################

export python=/home/liangzi/anaconda3/envs/HE/bin/python3
export root_dir="/home/liangzi/mpcgen/nlg/"

# export method="vanilla"
# export method="er"
# export p=0.0
# export eps=0.3

# export p_ls=(0.0 0.15 0.35 0.5 0.7 0.85)
export p_ls=(0.0 0.25 0.5 0.75)
export eps_ls=(0.0 0.15 0.35 0.5 0.7 0.85)
export method_ls=("vanilla" "er")

export cuda_num=5


for p in ${p_ls[*]};
do
    for eps in ${eps_ls[*]};
    do
	for method in ${method_ls[*]};
	do

echo "->RUNNING method ${method} with mask p ${p} and noise threshold eps${eps} on cuda:${cuda_num}"

$python robustness_exper.py\
	--method $method\
	--mask_p $p\
	--eps $eps\
	--cuda_num $cuda_num

	done
    done
done


# echo "->RUNNING method ${method} with mask p ${p} and noise threshold eps${eps} on cuda:${cuda_num}"
# $python robustness_exper.py\
# 	--method $method\
# 	--mask_p $p\
# 	--eps $eps\
# 	--cuda_num $cuda_num


echo "RUNNING 4.robustness_ananlysis.sh DONE."
# 4.robustness_ananlysis.sh ends here
