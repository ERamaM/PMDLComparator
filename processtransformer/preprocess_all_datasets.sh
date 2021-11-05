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
  for j in $(ls ./datasets/$i/ | grep "train" ); do
    real_fold=${j/"train_"/}
	  TMPDIR=$DUMP_DIR TS_SOCKET=/tmp/zarahah_preprocess $TS_EXECUTABLE python data_processing.py --raw_log_file ./datasets/$i/$real_fold --task next_activity --dir_path ./datasets/$i/ --dataset $real_fold
  done
done
