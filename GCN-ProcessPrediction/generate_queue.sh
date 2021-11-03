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

for i in $(ls ./data | grep  "train" | grep -v "nasa"); do
	  log=${i/train_/}
	  if [ $current_host == "ctgpgpu8" ] || [ $current_host == "ctgpgpu7" ]
	  then
	    CUDA_VISIBLE_DEVICES=1 TMPDIR=$DUMP_DIR TS_SOCKET=/tmp/venugopal $TS_EXECUTABLE python main.py --dataset $log
	  else
	    TS_SOCKET=/tmp/venugopal $TS_EXECUTABLE python main.py --dataset $log
	  fi
done
