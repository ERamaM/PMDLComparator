#!/bin/bash
rm results/*
rm output_models/*
rm best_models/*
for i in $(ls logs | grep -v "train" | grep -v "val" | grep -v "test"); do
  tsp java -jar splitminer_cmd-1.0.0-all.jar -l logs/"train_"$i -m output_models -b best_models
	tsp python run_dreamnap.py --dataset data/"$i" --train --test
done
