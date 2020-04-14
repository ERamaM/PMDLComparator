
echo "Test bpi_2012_w_complete.csv"
python evaluate_next_activity_and_time.py bpi_2012_w_complete.csv best_model.h5
echo "Test BPI_Challenge_2012_W_Complete.csv"
python evaluate_next_activity_and_time.py BPI_Challenge_2012_W_Complete.csv best_model.h5
echo "Test helpdesk.csv"
python evaluate_next_activity_and_time.py helpdesk.csv best_model.h5
echo "Test Helpdesk.csv"
python evaluate_next_activity_and_time.py Helpdesk.csv best_model.h5
