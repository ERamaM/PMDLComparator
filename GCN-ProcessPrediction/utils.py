import warnings

import pandas as pd
import time
from datetime import datetime
import numpy as np
from scipy import sparse as sp
from numpy import vstack
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.linalg import fractional_matrix_power

import torch
import torch.nn as nn
from torch import default_generator
from torch._utils import _accumulate
from torch.nn import Parameter
# from torch_geometric.nn.inits import glorot, zeros

from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_algorithm
from pm4py.objects.conversion.dfg import converter as dfg_conv
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.dfg import visualizer as dfg_vis_fact
from pm4py.visualization.petrinet import visualizer as pn_vis

def generate_features(df, total_activities, num_features):
    lastcase = ''
    firstLine = True
    numlines = 0
    casestarttime = None
    lasteventtime = None
    features = []

    for i, row in df.iterrows():
        t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
        if row[0] != lastcase:
            casestarttime = t
            lasteventtime = t
            lastcase = row[0]
            numlines += 1
        timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(lasteventtime))
        timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(casestarttime))
        midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = datetime.fromtimestamp(time.mktime(t)) - midnight
        timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
        timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
        timediff3 = timesincemidnight.seconds  # this leaves only time even occured after midnight
        timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday()  # day of the week
        lasteventtime = t
        firstLine = False
        feature_list = [timediff, timediff2, timediff3, timediff4]
        features.append(feature_list)

    df['Feature Vector'] = features

    firstLine = True
    NN_features = []

    for i, row in df.iterrows():
        if firstLine:
            features = np.zeros((total_activities, num_features))
            features[row[1] - 1] = row[3]
            firstLine = False
        else:
            if (row[3][0] == 0):
                features = np.zeros((total_activities, num_features))
                features[row[1] - 1] = row[3]
            else:
                features = np.copy(prev_row_features)
                features[row[1] - 1] = row[3]
        prev_row_features = features
        NN_features.append(features)

    return NN_features


def generate_labels(df, total_activities):
    next_activity = []
    next_timestamp = []

    for i, row in df.iterrows():
        if (i != 0):
            if (row[3][0] == 0):
                next_activity.append(total_activities)
            else:
                next_activity.append(row[1] - 1)
    next_activity.append(total_activities)
    for i, row in df.iterrows():
        if (i != 0):
            if (row[3][0] == 0):
                next_timestamp.append(0)
            else:
                next_timestamp.append(row[3][0])
    next_timestamp.append(0)

    return next_activity, next_timestamp


class EventLogData(Dataset):
    def __init__(self, input, output):
        self.X = input
        self.y = output
        self.y = self.y.to(torch.float32)
        self.y = self.y.reshape((len(self.y), 1))

    # get the number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at a particular index in the dataset
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get the indices for the train and test rows
    def get_splits(self, n_test=0.33, n_valid=0.2):
        train_idx, test_idx = train_test_split(list(range(len(self.X))), test_size=n_test, shuffle=False)
        train_idx, valid_idx = train_test_split(train_idx, test_size=n_valid, shuffle=True)
        train = Subset(self, train_idx)
        valid = Subset(self, valid_idx)
        test = Subset(self, test_idx)
        return train, valid, test


def prepare_data_for_Predictor(NN_features, label):
    dataset = EventLogData(NN_features, label)
    dataset_dl = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataset_dl


def generate_input_and_labels(train_path, val_path, test_path, num_nodes, num_features):
    total_unique_activities = num_nodes

    train_df = pd.read_csv(train_path)
    train_NN_features = generate_features(train_df, total_unique_activities, num_features)
    train_next_activity, train_next_timestamp = generate_labels(train_df, total_unique_activities)
    train_NN_features = torch.Tensor(train_NN_features).to(torch.float32)
    train_next_activity = torch.Tensor(train_next_activity).to(torch.float32)
    train_dl = prepare_data_for_Predictor(train_NN_features, train_next_activity)

    val_df = pd.read_csv(val_path)
    val_NN_features = generate_features(val_df, total_unique_activities, num_features)
    val_next_activity, val_next_timestamp = generate_labels(val_df, total_unique_activities)
    val_NN_features = torch.Tensor(val_NN_features).to(torch.float32)
    val_next_activity = torch.Tensor(val_next_activity).to(torch.float32)
    val_dl = prepare_data_for_Predictor(val_NN_features, val_next_activity)

    test_df = pd.read_csv(test_path)
    test_NN_features = generate_features(test_df, total_unique_activities, num_features)
    test_next_activity, test_next_timestamp = generate_labels(test_df, total_unique_activities)
    test_NN_features = torch.Tensor(test_NN_features).to(torch.float32)
    test_next_activity = torch.Tensor(test_next_activity).to(torch.float32)
    test_dl = prepare_data_for_Predictor(test_NN_features, test_next_activity)

    return train_dl, val_dl, test_dl


# Much better function than using "fractional_matrix_power"
# From here: https://github.com/danielegrattarola/spektral/blob/31ddf0770f9dc94337ea4e8b0fa284783496b42c/spektral/utils/convolution.py#L25
# Prevents nan in some graphs
def degree_power(A, k):
    r"""
    Computes \(\D^{k}\) from the given adjacency matrix. Useful for computing
    normalised Laplacian.
    :param A: rank 2 array or sparse matrix.
    :param k: exponent to which elevate the degree matrix.
    :return: if A is a dense array, a dense array; if A is sparse, a sparse
    matrix in DIA format.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        degrees = np.power(np.array(A.sum(1)), k).ravel()
    degrees[np.isinf(degrees)] = 0.0
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D

def generate_process_graph(path, num_nodes, binary_adjacency=True, laplacian_matrix=False):
    data = pd.read_csv(path)
    cols = ['case:concept:name', 'concept:name', 'time:timestamp']
    data.columns = cols
    data['time:timestamp'] = pd.to_datetime(data['time:timestamp'])
    data['concept:name'] = data['concept:name'].astype(str)
    log = log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)
    dfg = dfg_algorithm.apply(log)
    max = 0
    min = 0
    adj = np.zeros((num_nodes, num_nodes))
    for k, v in dfg.items():
        for i in range(num_nodes):
            if k[0] == str(i + 1):
                for j in range(num_nodes):
                    if k[1] == str(j + 1):
                        adj[i][j] = v
                        if v > max: max = v
                        if v < min: min = v

    # print("Raw weighted adjacencyFalse matrix: {}".format(adj))

    if binary_adjacency:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if (adj[i][j] != 0):
                    adj[i][j] = 1
        # print("Binary adjacency matrix: {}".format(adj))

    D = np.array(np.sum(adj, axis=1))
    D = np.matrix(np.diag(D))
    # print("Degree matrix: {}".format(D))

    adj = np.matrix(adj)

    if laplacian_matrix:
        adj = D - adj  # Laplacian Transform
        # print("Laplacian matrix: {}".format(adj))

    # adj = (D**-1)*adj
    # Use degree_power instead of fractional_power_matrix to avoid nans during the calculus of the adj matrix.
    adj = degree_power(D, -0.5) * adj * degree_power(D, -0.5)
    adj = torch.Tensor(adj).to(torch.float)

    # print("Symmetrically normalised Adjacency matrix: {}".format(adj))

    return adj


def visualize_process_graph(dfg, log):
    dfg_gv = dfg_vis_fact.apply(dfg, log, parameters={dfg_vis_fact.Variants.FREQUENCY.value.Parameters.FORMAT: "jpeg"})
    dfg_vis_fact.view(dfg_gv)
    dfg_vis_fact.save(dfg_gv, "dfg.jpg")


"""# Building Model"""

import math
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class GCNConv(torch.nn.Module):
    def __init__(self, num_nodes, num_features, out_channels):
        super(GCNConv, self).__init__()

        self.in_channels = num_features
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(num_features, out_channels))
        self.bias = Parameter(torch.Tensor(num_nodes))

        self.reset_parameters()

    def reset_parameters(self):
        # A.k.a glorot init
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, adj):
        x = adj @ x @ self.weight
        x = torch.flatten(x)
        x = x + self.bias
        return x


class EventPredictor(torch.nn.Module):
    def __init__(self, num_nodes, num_features=4):
        super(EventPredictor, self).__init__()

        self.layer1 = GCNConv(num_nodes, num_features, out_channels=1)
        self.layer2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_nodes, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, num_nodes + 1),
        )

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.layer2(x)
        # x = torch.sigmoid(x)
        return x

class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths, generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)]
