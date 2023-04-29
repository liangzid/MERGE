#!/bin/bash
######################################################################
#VARY_Parameters --- 

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created: 29 四月 2023
######################################################################

######################## Commentary ##################################
##  Varying Parameters
######################################################################

export python=/home/liangzi/anaconda3/envs/HE/bin/python3
# export root_dir="/home/liangzi/mpcGen/benchmark/"
export root_dir="/home/liangzi/mpcgen/benchmark/"

echo "================================"
echo "Evaluate for T5-base"
echo "================================"

export msl=128
export prefix=4

# s m 1.5m l xl xxl
# 82 180 265 391 1789 5591
export layer_ls=(6 12 12 24 48 48)
export d_ls=(512 768 1024 1024 2048 4096)
export head_ls=(8 12 16 16 32 64)

export methods=("vanillaGPT" "mpcformer_sfrelu" "mpcformer_sfquad" \
	      "thex" "onlyER" "our")
export gen_ls=("vanilla" "vanilla" "vanilla" "vanilla" "embedReSend"\
	      "embedReSend")
export sl=128

for j in `seq 1 ${#methods[@]}`;
do
    export layer=${layer_ls[$j-1]}
    export d=${d_ls[$j-1]}
    export head=${head_ls[$j-1]}
    
    for i in `seq 1 ${#methods[@]}`;
    do
	export method=${methods[$i-1]}
	export gen_type=${gen_ls[$i-1]}
	echo "method: $method; gen type: $gen_type"
	echo ">>>max sequence legnth: $sl<<<"

	export device=0
	export device2=0
	export port="394${device}"
	# export CUDA_VISIBLE_DEVICES="${device},${device2}"
	export CUDA_VISIBLE_DEVICES="${device}"
	export method="our"
	export gen_type="embedReSend"

	nohup $python profile_encdrdecdr.py 1 0\
	    $layer $d $sl $prefix $head $method $gen_type $port\
	    >./time_varyParams/ENCODER_DEC_method${method}gen_type${gen_type}$device$layer$d$msl$prefix$head$port.txt &

	$python profile_encdrdecdr.py 0 0\
		$layer $d $sl $prefix $head $method $gen_type $port 
    done
done


echo "RUNNING vary_params.sh DONE."
# vary_msl.sh ends here
