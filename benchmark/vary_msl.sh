#!/bin/bash
######################################################################
#VARY_MSL --- 

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created: 29 四月 2023
######################################################################

######################### Commentary ##################################
##  Varying Sequence Length
######################################################################

export python=/home/liangzi/anaconda3/envs/HE/bin/python3
# export root_dir="/home/liangzi/mpcGen/benchmark/"
export root_dir="/home/liangzi/mpcgen/benchmark/"

echo "================================"
echo "Evaluate for T5-base"
echo "================================"

export layer=12
export d=768
# export msl=256
# export msl=128
export msl=128
export prefix=4
export head=12
export methods=("vanillaGPT" "mpcformer_sfrelu" "mpcformer_sfquad" \
	      "thex" "onlyER" "our")
export gen_ls=("vanilla" "vanilla" "vanilla" "vanilla" "embedReSend"\
	      "embedReSend")
export msl_ls=(512 1024 2048)

for sl in ${msl_ls[*]};
do
    for i in `seq 1 ${#methods[@]}`;
    do
	export method=${methods[$i-1]}
	export gen_type=${gen_ls[$i-1]}
	echo "method: $method; gen type: $gen_type"
	echo ">>>max sequence legnth: $sl<<<"

	export device=4
	export device2=5
	export port="394${device}"
	export CUDA_VISIBLE_DEVICES="${device},${device2}"
	# export method="our"
	# export gen_type="embedReSend"

	nohup $python profile_encdrdecdr.py 1 1\
	    $layer $d $sl $prefix $head $method $gen_type $port\
	    >./time_varyMSL/ENCODER_DEC_method${method}gen_type${gen_type}$device$layer$d$msl$prefix$head$port.txt &

	$python profile_encdrdecdr.py 0 0\
		$layer $d $sl $prefix $head $method $gen_type $port 
    done
done


echo "RUNNING vary_msl.sh DONE."
# vary_msl.sh ends here
