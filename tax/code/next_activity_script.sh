#!/bin/bash
echo "" > aux_scripts/test_next.sh
echo "" > aux_scripts/test_suffix.sh
echo "" > aux_scripts/test_accuracy.sh
for i in `ls ./output_files/models/preprocessed`
do
	best_model=`ls ./output_files/models/preprocessed/$i | sort -V | tail -n 1`
	echo "echo \"Test $i\"" >> aux_scripts/test_next.sh
	echo "python evaluate_next_activity_and_time.py $i $best_model" >> aux_scripts/test_next.sh
	echo "python evaluate_suffix_and_remaining_time.py $i $best_model" >> aux_scripts/test_suffix.sh
	echo "python calculate_accuracy_on_next_event.py $i" >> aux_scripts/test_accuracy.sh
done

