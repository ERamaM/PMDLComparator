#!/bin/bash
for i in $(ls data | grep -v "train" | grep -v "val" | grep -v "test"); do
	tsp python deepm_act.py data/"$i" ACT results/"${$i%%.*}_results.log"
done
