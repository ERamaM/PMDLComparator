#!/bin/bash

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
if command -v ts &> /dev/null
then
    TS_EXECUTABLE=ts
else
    TS_EXECUTABLE=tsp
fi

current_host=`hostname`
DUMP_DIR=/home/efren.rama/tmp_hot_garbage/

# Prepare to dump tmp files in other folder.
if [ $current_host == "ctgpgpu8" ] || [ $current_host == "ctgpgpu7" ]
then
  mkdir -p $DUMP_DIR
fi

for i in $(ls testdata | grep -v "train" | grep -v "val" | grep -v "test"); do
  log=${i/.json/}
  python generate_configuration_log.py --dataset $i
done

conda activate hinkka
for i in $(ls testdata | grep -v "train" | grep -v "val" | grep -v "test"); do
	if [ $current_host == "ctgpgpu8" ] || [ $current_host == "ctgpgpu7" ]
	then
  TMPDIR=$DUMP_DIR CUDA_VISIBLE_DEVICES=0 TS_SOCKET=/tmp/hinkka $TS_EXECUTABLE python main.py -c config/custom_"$i"
  else
    TS_SOCKET=/tmp/hinkka $TS_EXECUTABLE python main.py -c config/custom_"$i"
  fi
done

