#!/bin/bash
##rm results/*
#rm output_models/*
#rm best_models/*
N_THREADS=8

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

for i in $(ls logs | grep "train" | grep -v "val"); do
	  log=${i/train_/}
	  full_log=${i/train_fold[0-9]_variation[0-9]_/}
	  train_val_log="train_val_"$log
  	TS_SOCKET=/tmp/theis $TS_EXECUTABLE python run_splitminer_new_version.py --log ./logs/$train_val_log --output_folder output_models --best_model best_models --n_threads $N_THREADS
	  #if [ $current_host == "ctgpgpu8" ] || [ $current_host == "ctgpgpu7" ]
	  #then
	  #  TMPDIR=$DUMP_DIR CUDA_VISIBLE_DEVICES=0 TS_SOCKET=/tmp/theis $TS_EXECUTABLE python run_dreamnap.py --fold_dataset logs/$log--full_dataset logs/$full_log --train --test
	  #  TMPDIR=$DUMP_DIR CUDA_VISIBLE_DEVICES=0 TS_SOCKET=/tmp/theis $TS_EXECUTABLE python run_dreamnap_no_resources.py --fold_dataset logs/$log --full_dataset logs/$full_log --train --test
	  #else
	  #  TS_SOCKET=/tmp/theis $TS_EXECUTABLE python run_dreamnap.py --fold_dataset logs/$log --full_dataset logs/$full_log --train --test
	  #  TS_SOCKET=/tmp/theis $TS_EXECUTABLE python run_dreamnap_no_resources.py --fold_dataset logs/$log --full_dataset logs/$full_log  --train --test
	  #fi
done
