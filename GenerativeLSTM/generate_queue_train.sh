#!/bin/bash
rm -r ./output_files/*
for i in $(ls input_files | grep -v "train" | grep -v "val" | grep -v "test"); do
  python experiment_generator.py --log input_files/$i --execute_inplace
done
