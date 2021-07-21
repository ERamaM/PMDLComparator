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

for i in $(ls ./data | grep  "train"); do
	  log=${i/train_/}
	  full_log=${i/train_fold[0-9]_variation[0-9]_/}
	  if [ $current_host == "ctgpgpu8" ] || [ $current_host == "ctgpgpu7" ]
	  then
	    CUDA_VISIBLE_DEVICES=1 TMPDIR=$DUMP_DIR TS_SOCKET=/tmp/thai_train $TS_EXECUTABLE python bpi_run.py --fold_dataset ./data/$log --full_dataset ./data/$full_log --train
	  else
	    TS_SOCKET=/tmp/thai_train $TS_EXECUTABLE python bpi_run.py --fold_dataset ./data/$log --full_dataset ./data/$full_log --train
	  fi
done
