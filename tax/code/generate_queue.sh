#!/bin/bash
rm results/*
if command -v ts &> /dev/null
then
    TS_EXECUTABLE=ts
else
    TS_EXECUTABLE=tsp
fi

for i in $(ls ../data | grep  "train"); do
	  log=${i/train_/}
	  full_log=${i/train_fold[0-9]_variation[0-9]_/}
	  TS_SOCKET=/tmp/tax $TS_EXECUTABLE python train.py --fold_dataset ../data/$log --full_dataset ../data/$full_log --train --test --test_suffix --test_suffix_calculus
done
