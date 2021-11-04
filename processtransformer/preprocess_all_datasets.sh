#!/bin/bash
for i in $(ls ./datasets | grep -v "nasa"); do
  for j in $(ls ./datasets/$i/ | grep "train" ); do
    real_fold=${j/"train_"/}
	  python data_processing.py --raw_log_file ./datasets/$i/$real_fold --task next_activity --dir_path ./datasets/$i/ --dataset $i
  done
done
