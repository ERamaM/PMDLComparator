#!/bin/bash
rm results/*
for i in $(ls data | grep -v "train" | grep -v "val" | grep -v "test"); do
	tsp python deeppm_act.py data/"$i" ACT results/"${i%%.*}_results.log"
done
