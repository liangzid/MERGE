#!/bin/bash
######################################################################
#EVALUATE_GPT ---

# Evaluate GPT-2 under private inference settings.

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created:  8 二月 2023
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

# echo ">> 1. evaluate our method vanilla"
# export method="our"
# export gen_type="vanilla"
# export device=1

# $python profile_gpt.py 0 $device\
# 	$layer $d $msl $prefix $head $method $gen_type &
	
# $python profile_gpt.py 1 $device\
# 	$layer $d $msl $prefix $head $method $gen_type >./res/ourvanilla.txt & 

echo ">> 2. evaluate mpcformer"
# export method="mpcformer_sfrelu"
export method="mpcformer_sfquad"
export method="our"
# export gen_type="vanilla"
export device=4
export port="3934"

$python profile_gpt.py 0 $device\
	$layer $d $msl $prefix $head $method $gen_type $port &
	
nohup $python profile_gpt.py 1 $device\
      $layer $d $msl $prefix $head $method $gen_type $port\
      >./res/$method$gen_type$device$layer$d$msl$prefix$head$port.txt &

# echo "================================"
# echo "Evaluate for vanilla GPT-2"
# echo "================================"

# export layer=12
# export d=768
# export msl=128
# export prefix=4
# export head=12

# echo ">> 1. evaluate MPCformer"
# export method="MPCformer"
# export gen_type="vanilla"
# export device=1

# $python profile_gpt.py 0 $device\
# 	$layer $d $msl $prefix $head $method $gen_type &
	
# $python profile_gpt.py 1 $device\
# 	$layer $d $msl $prefix $head $method $gen_type &
	
# echo ">> 2. evaluate Our + Vanilla Gen"
# export method="our"
# export gen_type="vanilla"
# export device=2

# $python profile_gpt.py 0 $device\
# 	$layer $d $msl $prefix $head $method $gen_type &
	
# $python profile_gpt.py 1 $device\
# 	$layer $d $msl $prefix $head $method $gen_type &
	
# echo ">> 3. evaluate Our + embedReSend-mode Gen"
# export method="our"
# export gen_type="embedReSend"
# export device=3

# $python profile_gpt.py 0 $device\
# 	$layer $d $msl $prefix $head $method $gen_type &
	
# $python profile_gpt.py 1 $device\
# 	$layer $d $msl $prefix $head $method $gen_type &






echo "RUNNING evaluate_gpt.sh DONE."
# evaluate_gpt.sh ends here
