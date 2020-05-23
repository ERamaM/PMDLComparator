#!/bin/bash
TSP=ts
rm -r ./output_files/*
for i in $(ls input_files | grep -v "train" | grep -v "val" | grep -v "test" | grep -v "embedded_matix"); do
  $TSP python evaluation_generator.py --log $i
done
