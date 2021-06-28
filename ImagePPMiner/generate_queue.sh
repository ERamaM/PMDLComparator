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

for i in $(ls dataset | grep "train"); do
	log=${i/train_/}
	full_log=${i/train_fold[0-9]_variation[0-9]_/}
  if [ $current_host == "ctgpgpu8" ] || [ $current_host == "ctgpgpu7" ]
  then
    TMPDIR=$DUMP_DIR CUDA_VISIBLE_DEVICES=0 TS_SOCKET=/tmp/pasquadibisceglie $TS_EXECUTABLE python run.py --fold_dataset dataset/$log --full_dataset dataset/$full_log --train --test
  else
    TS_SOCKET=/tmp/pasquadibisceglie $TS_EXECUTABLE python run.py --fold_dataset dataset/$log --full_dataset dataset/$full_log --train --test
  fi
done
