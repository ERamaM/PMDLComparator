import csv
from sklearn.metrics import accuracy_score
import argparse

parser = argparse.ArgumentParser(description="SeqPred")
parser.add_argument("eventlog", type=str)
args = parser.parse_args()
eventlog = args.eventlog

dataset = args.eventlog

ground_truth = []
predicted = []
with open("output_files/results/next_activity_and_time_"+dataset) as file:
    reader = csv.reader(file, delimiter=",")
    # Skip the header
    next(reader, None)
    for row in reader:
        ground_truth.append(row[2])
        predicted.append(row[3])

accuracy = accuracy_score(ground_truth, predicted)
print("Next activity accuracy: ", accuracy)
