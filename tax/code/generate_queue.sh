#!/bin/bash
rm results/*
for i in $(ls ../data | grep -v "train" | grep -v "val" | grep -v "test"); do
	tsp python train.py --dataset ../data/"$i" --train --test --test_suffix --test_suffix_calculus
done
