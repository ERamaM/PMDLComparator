#!/bin/bash
if command -v ts &> /dev/null
then
    TS_EXECUTABLE=ts
else
    TS_EXECUTABLE=tsp
fi

current_host=`hostname`
DUMP_DIR=/home/efren.rama/tmp_hot_garbage/

# Prepare to dump tmp files in other folder.
if [ $current_host == "ctgpgpu8" ] || [ $current_host == "ctgpgpu7" ]
then
  mkdir -p $DUMP_DIR
fi

for i in $(ls ./datasets | grep -v "nasa"); do
  for j in $(ls ./datasets/$i/ | grep "train"); do
    real_fold=${j/"train_"/}
	  if [ $current_host == "ctgpgpu8" ] || [ $current_host == "ctgpgpu7" ]
	  then
	    CUDA_VISIBLE_DEVICES=1 TMPDIR=$DUMP_DIR TS_SOCKET=/tmp/zarahah $TS_EXECUTABLE python next_activity.py --dataset $real_fold --epoch 100 --learning_rate 0.001
	  else
	    TS_SOCKET=/tmp/zarahah $TS_EXECUTABLE python next_activity.py --dataset $real_fold --epoch 100 --learning_rate 0.001
	  fi
	  done
done
