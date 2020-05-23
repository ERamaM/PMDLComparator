#!/bin/bash
TSP=ts
rm -r ./output_files/*
for i in $(ls input_files | grep -v "train" | grep -v "val" | grep -v "test"); do
  $TSP python evaluation_generator.py --log input_files/$i
done
