#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 11:59:43 2017

Test framework sources used to perform the tests required by paper: 
"hinkka"
by Markku Hinkka, Teemu Lehto and Keijo Heljanko
"""

import sys
import lasagne
from lasagne.layers import *
import numpy as np
import theano as theano
import theano.tensor as T
from time import time
import operator
import pickle
from my_utils import TraceData, writeLog, writeResultRow, get_filename, getOutputPath, OUTCOME_SELECTION_TOKEN_PREFIX, DURATION_TOKEN_PREFIX, EVENT_ATTRIBUTE_TOKEN_PREFIX, WORD_PART_SEPARATOR, CASE_ATTRIBUTE_TOKEN_PREFIX
#import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
from time import time
from pathlib import Path
import pandas as pd
import nltk
import itertools
import json

class Bucket:
    def __init__(self, num_layers, algorithm, num_units, hidden_dim_size, grad_clipping, optimizer, learning_rate):
        self.traces_train = []
        self.traces_test = []
        self.num_layers = num_layers
        self.algorithm = algorithm
        self.num_units = num_units
        self.hidden_dim_size = hidden_dim_size
        self.grad_clipping = grad_clipping
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        writeLog("Preparing " + str(self.num_layers) + " layers for algorithm: " + self.algorithm)

        # First, we build the network, starting with an input layer
        # Recurrent layers expect input of shape
        # (batch size, SEQ_LENGTH, num_features)
        mask_var = T.matrix('mask')

        l_in = lasagne.layers.InputLayer(shape=(None, None, num_units))
        l_mask = lasagne.layers.InputLayer((None, None), mask_var)
        self.l_layers = [l_in]

        # We now build the LSTM layer which takes l_in as the input layer
        # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients. 
        if (self.algorithm == "gru"):
            layerCreatorFunc = lambda parentLayer, isFirstLayer, isLastLayer: lasagne.layers.GRULayer(
                    parentLayer, self.hidden_dim_size, grad_clipping=self.grad_clipping,
                    mask_input = l_mask if isFirstLayer else None,
                    only_return_final=isLastLayer)
        else:
            # All gates have initializers for the input-to-gate and hidden state-to-gate
            # weight matrices, the cell-to-gate weight vector, the bias vector, and the nonlinearity.
            # The convention is that gates use the standard sigmoid nonlinearity,
            # which is the default for the Gate class.
#            gate_parameters = lasagne.layers.recurrent.Gate(
#                W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
#                b=lasagne.init.Constant(0.))
#            cell_parameters = lasagne.layers.recurrent.Gate(
#                W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
#                # Setting W_cell to None denotes that no cell connection will be used.
#                W_cell=None, b=lasagne.init.Constant(0.),
#                # By convention, the cell nonlinearity is tanh in an LSTM.
#                nonlinearity=lasagne.nonlinearities.tanh)

            layerCreatorFunc = lambda parentLayer, isFirstLayer, isLastLayer: lasagne.layers.LSTMLayer(
                    parentLayer, self.hidden_dim_size, grad_clipping=self.grad_clipping,
                    mask_input = l_mask if isFirstLayer else None,
                    nonlinearity=lasagne.nonlinearities.tanh,
                    # Here, we supply the gate parameters for each gate
#                    ingate=gate_parameters, forgetgate=gate_parameters,
#                    cell=cell_parameters, outgate=gate_parameters,
                    # We'll learn the initialization and use gradient clipping
                    only_return_final=isLastLayer)

        for layerId in range(self.num_layers):
            self.l_layers.append(layerCreatorFunc(self.l_layers[layerId], layerId == 0, layerId == self.num_layers - 1))

        # The output of l_forward_2 of shape (batch_size, N_HIDDEN) is then passed through the softmax nonlinearity to 
        # create probability distribution of the prediction
        # The output of this stage is (batch_size, vocab_size)
        self.l_out = lasagne.layers.DenseLayer(self.l_layers[len(self.l_layers) - 1], num_units=num_units, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)
        self.l_layers.append(self.l_out)
        
        # Theano tensor for the targets
        target_values = T.ivector('target_output')
#!        target_var = T.matrix('target_output')
    
        # lasagne.layers.get_output produces a variable for the output of the net
        network_output = lasagne.layers.get_output(self.l_out)

        # https://github.com/Lasagne/Lasagne/blob/master/examples/recurrent.py
        # The network output will have shape (n_batch, 1); let's flatten to get a
        # 1-dimensional vector of predicted values
#        predicted_values = network_output.flatten()

#        flat_target_values = target_values.flatten()

        # Our cost will be mean-squared error
#        cost = T.mean((predicted_values - flat_target_values)**2)
#        cost = T.mean((network_output - target_values)**2)
        # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
#!        cost = T.nnet.categorical_crossentropy(network_output,target_var).mean()
        cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(self.l_out,trainable=True)

        # Compute AdaGrad updates for training
        writeLog("Computing updates...")
        writeLog("Using optimizer: " + self.optimizer)
        if (self.optimizer == "sgd"):
            updates = lasagne.updates.sgd(cost, all_params, self.learning_rate)
        elif (self.optimizer == "adagrad"):
            updates = lasagne.updates.adagrad(cost, all_params, self.learning_rate)
        elif (self.optimizer == "adadelta"):
            updates = lasagne.updates.adagrad(cost, all_params, self.learning_rate, 0.95)
        elif (self.optimizer == "momentum"):
            updates = lasagne.updates.momentum(cost, all_params, self.learning_rate, 0.9)
        elif (self.optimizer == "nesterov_momentum"):
            updates = lasagne.updates.nesterov_momentum(cost, all_params, self.learning_rate, 0.9)
        elif (self.optimizer == "rmsprop"):
            updates = lasagne.updates.rmsprop(cost, all_params, self.learning_rate, 0.9)
        else:
            updates = lasagne.updates.adam(cost, all_params, self.learning_rate, beta1=0.9, beta2=0.999)

        # Theano functions for training and computing cost
        writeLog("Compiling train function...")
        self.train = theano.function([l_in.input_var, target_values, l_mask.input_var], cost, updates=updates, allow_input_downcast=True)
#!        self.train = theano.function([l_in.input_var, target_var, l_mask.input_var], cost, updates=updates, allow_input_downcast=True)
        writeLog("Compiling train cost computing function...")
#        self.compute_cost = theano.function([l_in.input_var, target_values, l_mask.input_var], cost, allow_input_downcast=True)

        # In order to generate text from the network, we need the probability distribution of the next character given
        # the state of the network and the input (a seed).
        # In order to produce the probability distribution of the prediction, we compile a function called probs. 
        writeLog("Compiling propabilities computing function...")
        self.propabilities = theano.function([l_in.input_var, l_mask.input_var],network_output,allow_input_downcast=True)
