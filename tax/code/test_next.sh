#!/bin/bash
echo "Test bpi_2012_a.csv"
python evaluate_next_activity_and_time.py bpi_2012_a.csv model_08-1.19.h5
echo "Test bpi_2012_complete.csv"
python evaluate_next_activity_and_time.py bpi_2012_complete.csv model_65-1.40.h5
echo "Test bpi_2012.csv"
python evaluate_next_activity_and_time.py bpi_2012.csv model_53-1.43.h5
echo "Test bpi_2012_o.csv"
python evaluate_next_activity_and_time.py bpi_2012_o.csv model_75-1.06.h5
echo "Test bpi_2012_w_complete.csv"
python evaluate_next_activity_and_time.py bpi_2012_w_complete.csv model_31-1.67.h5
echo "Test bpi_2012_w.csv"
python evaluate_next_activity_and_time.py bpi_2012_w.csv model_81-1.33.h5
echo "Test bpi_2013_incidents.csv"
python evaluate_next_activity_and_time.py bpi_2013_incidents.csv model_60-0.95.h5
echo "Test helpdesk.csv"
python evaluate_next_activity_and_time.py helpdesk.csv model_70-0.94.h5
echo "Test sepsis.csv"
python evaluate_next_activity_and_time.py sepsis.csv model_95-1.54.h5
