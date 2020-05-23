#!/bin/bash
rm -r ./output_files/*
N_SLOTS=3
for i in $(ls input_files | grep -v "train" | grep -v "val" | grep -v "test" | grep -v "embedded_matix"); do
  python experiment_generator.py --log input_files/$i --execute_inplace --slots $N_SLOTS
done
