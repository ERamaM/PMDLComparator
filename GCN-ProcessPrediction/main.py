# -*- coding: utf-8 -*-
"""Copy of GCN_EventPredictor_(Training).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iKULo9ib0KmoDqWwoC5oO8Xi1XJqqgKq

# Importing necessary packages and functions
"""

# !pip install pm4py

print("Starting up")

from copy import copy
import re
import pandas as pd
from sklearn.metrics import accuracy_score
from utils import EventPredictor, generate_process_graph, generate_input_and_labels
import argparse
import torch
import torch.nn as nn
from pm4py.objects.conversion.log import converter as log_converter



"""# Setting the parameters"""

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Name of the dataset (e.g: fold0_variation0_Helpdesk.csv)", required=True)
args = parser.parse_args()


fold_log = args.dataset
print("Fold log: ", fold_log)
full_log = re.sub("fold\\d_variation\\d_", "", fold_log)
dataset = re.sub(".xes.gz", "", fold_log)
print("Full log: ", full_log)
full_path = "./data/" + full_log
train_path = "./data/train_" + fold_log
val_path = "./data/val_" + fold_log
test_path = "./data/test_" + fold_log

print("Full path: ", full_path)
print("Train path: ", train_path)


# path = '/content/drive/My Drive/MSc Dissertation/Data/helpdesk.csv'
# save_folder = '/content/Results/helpdesk'

num_features = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
num_epochs = 100
seed_value = 42
# lr_value = 1e-05
weighted_adjacency = True
binary_adjacency = False
laplacian_matrix = False
variant = 'weighted'  # Choose from ['weighted','binary','laplacianOnWeighted','laplacianOnBinary']
num_runs = 1

def count_nodes(path):
    data = pd.read_csv(path)
    data_cpy = copy(data)
    data_cpy.columns = ["case:concept:name", "concept:name", "time:timestamp"]
    log = log_converter.apply(data_cpy, variant=log_converter.Variants.TO_EVENT_LOG)
    n_nodes = len(set([event["concept:name"] for trace in log for event in trace]))
    return n_nodes

lr_run = 1
for lr_run in range(lr_run, 2):
    if lr_run == 0:
        lr_value = 1e-03
    elif lr_run == 1:
        lr_value = 1e-04
    elif lr_run == 2:
        lr_value = 1e-05
    run = 0
    for run in range(num_runs):
        print("Run: {}, Learning Rate: {}".format(run + 1, lr_value))
        # Count the activities in the whole log since some can appear in the test set and not in the training one
        num_nodes = count_nodes(full_path)
        model = EventPredictor(num_nodes, num_features)
        adj = generate_process_graph(train_path, num_nodes)
        train_dl, valid_dl, test_dl = generate_input_and_labels(train_path, val_path, test_path, num_nodes, num_features)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_value)
        # print("************* Event Predictor ***************")
        # print("Train size: {}, Validation size:{}, Test size: {}".format(len(train_dl.dataset),len(valid_dl.dataset),len(test_dl.dataset)))
        # print(model)
        model = model.to(device)
        adj = adj.to(device)
        epochs_plt = []
        acc_plt = []
        loss_plt = []
        valid_loss_plt = []

        for epoch in range(num_epochs):

            model.train()
            num_train = 0
            training_loss = 0
            predictions, actuals = list(), list()

            for i, (inputs, targets) in enumerate(train_dl):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()  # Clearing the gradients

                yhat = model(inputs[0], adj)

                loss = criterion(yhat.reshape((1, -1)), targets[0].to(torch.long))
                loss.backward()
                optimizer.step()

                training_loss += loss.item()

                yhat = yhat.to('cpu')
                yhat = torch.argmax(yhat)
                actual = targets.to('cpu')
                actual = actual[0]
                predictions.append(yhat)
                actuals.append(actual)
                num_train += 1

            with torch.no_grad():
                model.eval()
                num_valid = 0
                validation_loss = 0
                for i, (inputs, targets) in enumerate(valid_dl):
                    inputs, targets = inputs.to(device), targets.to(device)
                    yhat_valid = model(inputs[0], adj)
                    loss_valid = criterion(yhat_valid.reshape((1, -1)), targets[0].to(torch.long))
                    validation_loss += loss_valid.item()
                    num_valid += 1

            acc = accuracy_score(actuals, predictions)
            avg_training_loss = training_loss / num_train
            avg_validation_loss = validation_loss / num_valid

            if (epoch == 0):
                best_loss = avg_validation_loss
                torch.save(model.state_dict(),
                           '{}/EventPredictor_parameters_{}_{}_{}_run{}.pt'.format("results/", dataset, variant,
                                                                                   lr_value, run))

            if (avg_validation_loss < best_loss):
                torch.save(model.state_dict(),
                           '{}/EventPredictor_parameters_{}_{}_{}_run{}.pt'.format("results/", dataset, variant,
                                                                                   lr_value, run))
                best_loss = avg_validation_loss

            print("Epoch: {}, Loss: {}, Accuracy: {}, Validation loss : {}".format(epoch, avg_training_loss, acc,
                                                                                   avg_validation_loss))
            epochs_plt.append(epoch + 1)
            acc_plt.append(acc)
            loss_plt.append(avg_training_loss)
            valid_loss_plt.append(avg_validation_loss)

        filepath = '{}/Accuracy_{}_{}_{}_run{}.txt'.format("results/", dataset, variant, lr_value, run)
        with torch.no_grad():
            model.eval()
            num_valid = 0
            validation_loss = 0
            for i, (inputs, targets) in enumerate(test_dl):
                inputs, targets = inputs.to(device), targets.to(device)
                yhat_valid = model(inputs[0], adj)
                loss_valid = criterion(yhat_valid.reshape((1, -1)), targets[0].to(torch.long))
                validation_loss += loss_valid.item()
                num_valid += 1

        acc = accuracy_score(actuals, predictions)
        with open(filepath, "w") as result_file:
            result_file.write("Test accuracy: " + str(acc))


