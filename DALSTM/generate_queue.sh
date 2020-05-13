#!/bin/bash
rm results/*
for i in $(ls data | grep -v "train" | grep -v "val" | grep -v "test"); do
	tsp python LSTM_sequence_mae.py --dataset data/$i --train --test
done
