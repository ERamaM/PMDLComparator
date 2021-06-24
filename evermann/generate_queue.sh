#!/bin/bash
rm results/*
# Find task-spooler executable
if command -v ts &> /dev/null
then
    TS_EXECUTABLE=ts
else
    TS_EXECUTABLE=tsp
fi

for i in $(ls data | grep  "train"); do
	  #tsp python train.py --dataset data/$i --train --test --test_suffix
	  log=${i/train_/}
	  full_log=${i/train_fold[0-9]_variation[0-9]_/}
	  CUDA_VISIBLE_DEVICES=0 TS_SOCKET=/tmp/evermann $TS_EXECUTABLE python train.py --fold_dataset data/$log --full_dataset data/$full_log --train --test --test_suffix --test_suffix_calculus
done
