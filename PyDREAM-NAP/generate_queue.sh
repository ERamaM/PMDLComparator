#!/bin/bash
rm results/*
rm output_models/*
rm best_models/*
N_THREADS=8
for i in $(ls logs | grep -v "train" | grep -v "val" | grep -v "test"); do
  tsp python run_splitminer.py --log ./logs/"train_val_"$i --output_folder output_models --best_model best_models --n_threads $N_THREADS
	tsp python run_dreamnap.py --dataset logs/"$i" --train --test
done
