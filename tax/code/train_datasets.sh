#!/bin/bash

echo "" > train_script.sh
for dataset in `ls ../data/preprocessed/`
do
	echo "Training: $dataset"
	echo "python train.py "../data/preprocessed/$dataset"" >> aux_scripts/train_script.sh
done
