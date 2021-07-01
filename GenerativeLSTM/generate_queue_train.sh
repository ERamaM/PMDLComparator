#!/bin/bash
# rm -r ./output_files/*
N_SLOTS=1
if command -v ts &> /dev/null
then
    TS_EXECUTABLE=ts
else
    TS_EXECUTABLE=tsp
fi

current_host=`hostname`
DUMP_DIR=/home/efren.rama/tmp_hot_garbage/
for i in $(ls input_files | grep -v "train" | grep -v "val" | grep -v "test" | grep -v "embedded_matix"); do
	  full_log=${i/train_fold[0-9]_variation[0-9]_/}
    TMPDIR=$DUMP_DIR CUDA_VISIBLE_DEVICES=0 TS_SOCKET=/tmp/camargo $TS_EXECUTABLE python lstm.py -a emb_training -f $full_log -o True
done
for i in $(ls input_files | grep "train"); do
	log=${i/train_/}
	full_log=${i/train_fold[0-9]_variation[0-9]_/}
  python experiment_generator.py --fold_log input_files/$log --full_log input_files/$full_log --execute_inplace --slots $N_SLOTS
done
