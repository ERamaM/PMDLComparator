#!/bin/bash
# rm -r ./output_files/*
N_SLOTS=1
for i in $(ls input_files | grep "train"); do
	log=${i/train_/}
	full_log=${i/train_fold[0-9]_variation[0-9]_/}
  python experiment_generator.py --fold_log input_files/$log --full_log input_files/$full_log --execute_inplace --slots $N_SLOTS
done
