#!/bin/bash
if command -v ts &> /dev/null
then
    TS_EXECUTABLE=ts
else
    TS_EXECUTABLE=tsp
fi
for i in $(ls input_files | grep  "train"); do
	log=${i/train_/}
  TS_SOCKET=/tmp/camargo $TS_EXECUTABLE  python evaluation_generator.py --log $log
done
