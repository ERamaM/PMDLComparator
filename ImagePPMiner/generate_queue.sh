#!/bin/bash
rm results/*
for i in $(ls dataset | grep -v "train" | grep -v "val" | grep -v "test"); do
	tsp python run.py --dataset dataset/$i
done
