#!/bin/bash
######################################################################
#TEST_MERGE --- 

# evaluate the time cost of MERGE, our methods.

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created: 18 四月 2023
######################################################################

export python=/home/liangzi/anaconda3/envs/HE/bin/python3
# export root_dir="/home/liangzi/mpcGen/benchmark/"
export root_dir="/home/liangzi/mpcgen/benchmark/"

export device_ls=(0 1 2 3 4 5 6 7)
export layer_ls=(12)
export d_ls=(768)
export msl_ls=(128 256 512)
export p_ls=(4 8 16)
export head_ls=(12)
export method_ls=("MPCformer" "our")
export genmethod_ls=("vanilla" "embedReSend")

echo "================================"
echo "Evaluate for vanilla GPT-2"
echo "================================"

export layer=12
export d=768
# export msl=256
export msl=128
export prefix=4
export head=12

echo ">> 2. evaluate only MM"
# export method="mpcformer_sfrelu"
# export method="mpcformer_sfquad"
# export method="thex"
# export method="vanillaGPT"
export method="onlyMM"
# export method="onlyER"
# export method="our"
# export gen_type="vanilla" 
export gen_type="embedReSend" 
export device=6
export port="394${device}"
export CUDA_VISIBLE_DEVICES="${device}"
export device=0

$python profile_gpt.py 0 $device\
	$layer $d $msl $prefix $head $method $gen_type $port &
	
nohup $python profile_gpt.py 1 $device\
      $layer $d $msl $prefix $head $method $gen_type $port\
      >./res/method${method}gen_type${gen_type}$device$layer$d$msl$prefix$head$port.txt &

echo "RUNNING test_merge.sh DONE."
# test_merge.sh ends here
