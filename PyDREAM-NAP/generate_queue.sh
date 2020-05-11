#!/bin/bash
rm results/*
rm output_models/*
rm best_models/*
N_THREADS=8
for i in $(ls logs | grep -v "train" | grep -v "val" | grep -v "test"); do
  tsp java -jar splitminer_cmd-1.0.0-all.jar -l logs/"train_val_"$i -m output_models -b best_models -t $N_THREADS
	tsp python run_dreamnap.py --dataset logs/"$i" --train --test
done
