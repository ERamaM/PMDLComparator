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
import traceback
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
from my_utils import writeLog, writeResultRow, get_filename, getOutputPath, ATTRIBUTE_COLUMN_PREFIX, OTHER_TOKEN, OUTCOME_SELECTION_TOKEN_PREFIX, DURATION_TOKEN_PREFIX, EVENT_ATTRIBUTE_TOKEN_PREFIX, WORD_PART_SEPARATOR, CASE_ATTRIBUTE_TOKEN_PREFIX, TRACE_FINISH_TOKEN
from bucket import Bucket

UNKNOWN_TOKEN = "UNKNOWN"
DURATION_VALUE_PLACEHOLDER_TOKEN = "__DURATION_VALUE__"

class Model:
    def __init__(self, parameters):
        writeLog("Creating new model object")
        self.parameters = parameters
        self.algorithm = parameters["algorithm"]
        self.num_layers = parameters["num_layers"]
        self.optimizer = parameters["optimizer"]
        self.learning_rate = parameters["learning_rate"]
        self.batch_size = parameters["batch_size"]
        self.num_callbacks = parameters["num_callbacks"]
        self.case_name = parameters["case_name"]
        self.hidden_dim_size = parameters["hidden_dim_size"]
        self.num_iterations_between_reports = parameters["num_iterations_between_reports"]
        self.grad_clipping = parameters["grad_clipping"]
        self.predict_only_outcome = parameters["predict_only_outcome"]
        self.final_trace_only = parameters["final_trace_only"]
        self.max_num_words = parameters["max_num_words"]
        self.trace_length_modifier = parameters["trace_length_modifier"]
        self.truncate_unknowns = parameters["truncate_unknowns"]
        self.eventlogActivities = None
        self.eventlogAttributes = None
        self.eventlogFilename = None
        self.eventlogFilepath = None

    def initialize(self, case_clustering, event_clustering, rng):
        self.case_clustering = case_clustering
        self.event_clustering = event_clustering
        self.rng = rng

    def train(self, eventlog):
        self.train_start_time = time()
        self.eventlog = eventlog
        self.eventlog.model = self
        try:
            self.eventlog.preProcessForTraining(self.parameters)
            self.initClusters(eventlog)
            self.prepareTestData(eventlog)
            self.traces_train = eventlog.convertTracesFromInputData(eventlog.trainingData, self.parameters, self.trace_length_modifier)
            self.traces_validation = eventlog.convertTracesFromInputData(eventlog.validationData, self.parameters, self.trace_length_modifier)
            self.traces_test = eventlog.convertTracesFromInputData(eventlog.testData, self.parameters, self.trace_length_modifier)

            maxNumTraces = self.parameters["max_num_traces_in_training"]
            if (maxNumTraces != None) and (maxNumTraces < len(self.traces_train)):
                writeLog("Filtering %d traces out of %d training traces" % (maxNumTraces, len(self.traces_train)))
                self.traces_train = list(np.random.choice(np.asarray(self.traces_train), maxNumTraces, replace=False))
            maxNumTraces = self.parameters["max_num_traces_in_training_test"]
            if (maxNumTraces != None) and (maxNumTraces < len(self.traces_test)):
                writeLog("Filtering %d traces out of %d validation traces" % (maxNumTraces, len(self.traces_test)))
                self.traces_test = list(np.random.choice(np.asarray(self.traces_test), maxNumTraces, replace=False))
            return self.createModel()
        except:
            writeLog("Exception: " + traceback.format_exc())
#            writeLog("Exception: " + sys.exc_info()[0])

    def prepareTestData(self, eventlog):
        # Cluster validation and test splits
        testData = eventlog.testData
        if (self.case_clustering != None):
            self.case_clustering.clusterCases(eventlog, eventlog.testData)
            self.case_clustering.clusterCases(eventlog, eventlog.validationData)
        else:
            for td in enumerate(testData):
                td["_cluster"] = 0
            for td in enumerate(eventlog.validationData):
                td["_cluster"] = 0

        if (self.event_clustering != None):
            self.event_clustering.clusterEvents(eventlog, eventlog.testData)
            self.event_clustering.clusterEvents(eventlog, eventlog.validationData)
        else:
            for c in testData:
                for e in c["t"]:
                    e.append(0)
            for c in eventlog.validationData:
                for e in c["t"]:
                    e.append(0)

    def initClusters(self, eventlog):
        self.case_clustering.trainForCaseClustering(eventlog, eventlog.trainingData)
        self.event_clustering.trainForEventClustering(eventlog, eventlog.trainingData)

    def gen_data(self, traces, p, positions, batch_size):
        '''
        This function produces a semi-redundant batch of training samples from the location 'p' in the provided string (data).
        For instance, assuming SEQ_LENGTH = 5 and p=0, the function would create batches of 
        5 characters of the string (starting from the 0th character and stepping by 1 for each semi-redundant batch)
        as the input and the next character as the target.
        To make this clear, let us look at a concrete example. Assume that SEQ_LENGTH = 5, p = 0 and BATCH_SIZE = 2
        If the input string was "The quick brown fox jumps over the lazy dog.",
        For the first data point,
        x (the inputs to the neural network) would correspond to the encoding of 'T','h','e',' ','q'
        y (the targets of the neural network) would be the encoding of 'u'
        For the second point,
        x (the inputs to the neural network) would correspond to the encoding of 'h','e',' ','q', 'u'
        y (the targets of the neural network) would be the encoding of 'i'
        The data points are then stacked (into a three-dimensional tensor of size (batch_size,SEQ_LENGTH,vocab_size))
        and returned. 
        Notice that there is overlap of characters between the batches (hence the name, semi-redundant batch).
        '''
        data_size = len(positions) if positions != None else len(traces)
        x = np.zeros((batch_size, self.seq_length, len(self.word_to_index)))
        y = np.zeros(batch_size)
        masks = []
        
        disableActivityLabels = self.parameters["disable_activity_labels"]
        disableDurations = self.parameters["disable_durations"]
        disableEventAttributes = self.parameters["disable_event_attributes"]
        disableCaseAttributes = self.parameters["disable_case_attributes"]
        disableRawEventAttributes = self.parameters["disable_raw_event_attributes"]
        disableRawCaseAttributes = self.parameters["disable_raw_case_attributes"]
        useSingleValueForDuration = self.parameters["use_single_value_for_duration"]
        for n in range(batch_size):
            ptr = (p + n) % data_size
            traceId = positions[ptr][0] if positions != None else ptr
            trace = traces[traceId]
            traceLastId = positions[ptr][1] if positions != None else len(traces[traceId].tokenized_sentences)
            caCluster = None if disableCaseAttributes else self.word_to_index[trace.caseAttributeCluster]
            caWords = None if disableRawCaseAttributes else trace.getCaseAttributeWordIndexes(self.word_to_index)
            eaWords = None if disableRawEventAttributes else trace.getEventAttributeWordIndexes(self.word_to_index)
            for i in range(traceLastId):
                if ((not disableActivityLabels)):
                    label = trace.activityLabels[i]
                    x[n, i, self.word_to_index[label]] = 1.
                if ((caCluster != None) and (not disableCaseAttributes)):
                    x[n, i, caCluster] = 1.
                duration = trace.durations[i]
                if (not disableDurations):
                    if useSingleValueForDuration:
                        x[n, i, self.word_to_index[DURATION_VALUE_PLACEHOLDER_TOKEN]] = trace.durationValues[i]
                    elif (duration != None):
                        x[n, i, self.word_to_index[duration]] = 1.
                cluster = trace.eventAttributeClusters[i]
                if ((cluster != None) and (not disableEventAttributes)):
                    x[n, i, self.word_to_index[cluster]] = 1.
                if (not disableRawCaseAttributes):
                    for w in caWords:
                        if w != None:
                            x[n, i, w] = 1.
                if (not disableRawEventAttributes):
                    for w in eaWords[i]:
                        if w != None:
                            x[n, i, w] = 1.

            masks.append([1 if x < traceLastId else 0 for x in range(self.seq_length)])
            y[n] = self.word_to_index[trace.outcomeToken] if self.predict_only_outcome else (self.word_to_index[trace.tokenized_sentences[traceLastId] if traceLastId < len(trace.activityLabels) else trace.outcomeToken])
        return x, np.array(y,dtype='int32'), np.asarray(masks)

    def gen_prediction_data(self, traces, tracePercentage):
        batches = []
        masks = []
        numTraces = len(traces)
        if (numTraces == 0):
            return np.asarray(batches), np.asarray(masks)
        batchRow = 0
        x = np.zeros((self.batch_size if (numTraces > self.batch_size) else numTraces, self.seq_length, len(self.word_to_index)))
        m = np.zeros((self.batch_size if (numTraces > self.batch_size) else numTraces, self.seq_length))
        batches.append(x)
        masks.append(m)

        disableActivityLabels = self.parameters["disable_activity_labels"]
        disableDurations = self.parameters["disable_durations"]
        disableEventAttributes = self.parameters["disable_event_attributes"]
        disableCaseAttributes = self.parameters["disable_case_attributes"]
        disableRawEventAttributes = self.parameters["disable_raw_event_attributes"]
        disableRawCaseAttributes = self.parameters["disable_raw_case_attributes"]
        useSingleValueForDuration = self.parameters["use_single_value_for_duration"]
        dpIndex = self.word_to_index[DURATION_VALUE_PLACEHOLDER_TOKEN] if useSingleValueForDuration and (not disableDurations) else None

        for traceRow in range(len(traces)):
            trace = traces[traceRow]
            (labels, durations, clusters, durationValues) = trace.getActivitiesForPrediction(self.word_to_index, tracePercentage, self.truncate_unknowns, self.seq_length, len(self.word_to_index), disableActivityLabels, disableDurations, disableEventAttributes, disableCaseAttributes, useSingleValueForDuration)
            caCluster = None if disableCaseAttributes else self.word_to_index[trace.caseAttributeCluster]
            caWords = None if disableRawCaseAttributes else trace.getCaseAttributeWordIndexes(self.word_to_index)
            eaWords = None if disableRawEventAttributes else trace.getEventAttributeWordIndexes(self.word_to_index)
            for i in range(min(len(labels), x.shape[1] - 1)):
                if ((not disableActivityLabels)):
                    x[batchRow, i, labels[i]] = 1.
                if ((caCluster != None) and (not disableCaseAttributes)):
                    x[batchRow, i, caCluster] = 1.
                if (not disableDurations):
                    duration = durations[i]
                    if useSingleValueForDuration:
                        x[batchRow, i, dpIndex] = durationValues[i]
                    elif (duration != None):
                        x[batchRow, i, duration] = 1.
                if ((clusters[i] != None) and (not disableEventAttributes)):
                    x[batchRow, i, clusters[i]] = 1.
                if (not disableRawCaseAttributes):
                    for w in caWords:
                        if w != None:
                            x[batchRow, i, w] = 1.
                if (not disableRawEventAttributes):
                    try: # For some reason this fails when calculating the validation accuracy
                        for w in eaWords[i]:
                            if w != None:
                                x[batchRow, i, w] = 1.
                    except:
                        pass
            for i in range(self.seq_length):
                m[batchRow, i] = 1 if i < len(labels) else 0
            batchRow += 1
            if (batchRow >= self.batch_size):
                x = np.zeros((self.batch_size if (numTraces - traceRow) > self.batch_size else (numTraces - traceRow - 1), self.seq_length, len(self.word_to_index)))
                m = np.zeros((self.batch_size if (numTraces - traceRow) > self.batch_size else (numTraces - traceRow - 1), self.seq_length))
                batches.append(x)
                masks.append(m)
                batchRow = 0
        return np.asarray(batches), np.asarray(masks)

    def trainModel(self, callback):
        writeLog("Training...")
        p = 0
        data_size = len(self.traces_train)
        if self.parameters["num_epochs_per_iteration"] != None:
            self.num_iterations_between_reports = self.parameters["num_epochs_per_iteration"] * data_size
        num_iterations = 0
        num_iterations_after_report = 0
        num_report_iterations = 1
        avg_cost = 0
#        writeLog("It: " + str(data_size * self.num_epochs // self.batch_size))
        try:
            it = 0
            while (num_report_iterations <= self.num_callbacks):
                x, y, mask = self.gen_data(self.traces_train, p, self.positions_train, self.batch_size)
                it += 1
                p += self.batch_size 
                num_iterations += self.batch_size
                num_iterations_after_report += self.batch_size
#                if(p+self.batch_size+self.seq_length >= data_size):
#                    writeLog('Carriage Return')
#                    p = 0;
                avg_cost += self.rnn_train(x, y, mask)
                if (callback and num_iterations_after_report >= self.num_iterations_between_reports):
                    callback(num_iterations, it, avg_cost / it, num_report_iterations)
                    avg_cost = 0
                    num_iterations_after_report = num_iterations_after_report - self.num_iterations_between_reports
                    num_report_iterations = num_report_iterations + 1

#            callback(num_iterations, it, avg_cost / it, num_report_iterations)
        except KeyboardInterrupt:
            pass

    def initializeTraces(self):
        word_to_index = []
        index_to_word = []

        # Tokenize the sentences into words
        writeLog("Tokenizing %s training- and %s test sentences." % (len(self.traces_train), len(self.traces_test)))
    #    tokenized_sentences = [nltk.word_tokenize(trace.sentence) for trace in traces]
#        tokenized_sentences_train = [nltk.WhitespaceTokenizer().tokenize(trace.sentence) for trace in self.traces_train]
        tokenized_sentences_train = [trace.activityLabels for trace in self.traces_train]
        durations = set()
        self.outcomes = set()
        predict_next_activity = self.parameters["predict_next_activity"]
        includeRawEventAttributes = not self.parameters["disable_raw_event_attributes"] and (self.event_clustering != None)
        includeRawCaseAttributes = not self.parameters["disable_raw_case_attributes"] and (self.case_clustering != None)

        rawCaseAttributes = []
        rawEventAttributes = []
        for trace in self.traces_train:
            durations.update(trace.durations)
            if (not predict_next_activity):
                self.outcomes.add(trace.outcomeToken)

            if (includeRawCaseAttributes):
                self.case_clustering.filterCaseAttributes(self.eventlog, trace)
            if (includeRawEventAttributes):
                self.event_clustering.filterEventAttributes(self.eventlog, trace)

        if (includeRawCaseAttributes):
            rawCaseAttributes = self.case_clustering.getSetOfKnownCaseAttributeValuesForVocabulary(self.eventlog, self.traces_train)
        if (includeRawEventAttributes):
            rawEventAttributes = self.event_clustering.getSetOfKnownEventAttributeValuesForVocabulary(self.eventlog, self.traces_train)

        if (predict_next_activity):
            self.outcomes.add(OUTCOME_SELECTION_TOKEN_PREFIX + TRACE_FINISH_TOKEN)
        durations = [d for d in durations if d != None]

        eventAttributeClusters = [EVENT_ATTRIBUTE_TOKEN_PREFIX + str(l) for l in self.event_clustering.getClusterLabels()]
        caseAttributeClusters = [CASE_ATTRIBUTE_TOKEN_PREFIX + str(l) for l in self.case_clustering.getClusterLabels()]

        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences_train))
        writeLog("Found %d unique words tokens." % len(word_freq.items()))

        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
        writeLog("Using vocabulary size %d." % len(vocab))
        writeLog("Words frequencies in the vocabulary:")

        words = []
        for x in vocab:
            w = x[0]
            writeLog("  %d * %s" % (x[1], x[0]))
            words.append(w)
        words = np.asarray(words)
        self.outcomes = list(self.outcomes)

        if ((self.max_num_words != None) and (self.max_num_words < len(words))):
            words = words[range(self.max_num_words)]
            writeLog("Vocabulary was truncated to %d most frequent words in training set." % len(words))

        disable_activity_labels = self.parameters["disable_activity_labels"]
        disable_durations = self.parameters["disable_durations"]
        disable_event_attributes = self.parameters["disable_event_attributes"]
        disable_case_attributes = self.parameters["disable_case_attributes"]
        create_unknown_token = self.parameters["create-unknown-tokens"]

        if disable_durations:
            durations = []
        if disable_event_attributes:
            eventAttributeClusters = []
        if disable_case_attributes:
            caseAttributeClusters = []
        if disable_activity_labels and (not predict_next_activity):
            words = []
        others = [UNKNOWN_TOKEN]
        if (not disable_event_attributes) and create_unknown_token:
            others.append(EVENT_ATTRIBUTE_TOKEN_PREFIX + "None")
        if (not disable_case_attributes) and create_unknown_token:
            others.append(CASE_ATTRIBUTE_TOKEN_PREFIX + "None")
        
        useSingleValueForDuration = self.parameters["use_single_value_for_duration"]
        if (not disable_durations) and self.parameters["use_single_value_for_duration"]:
            durations = [DURATION_VALUE_PLACEHOLDER_TOKEN]

        self.index_to_word = np.concatenate([self.outcomes, durations, eventAttributeClusters, caseAttributeClusters, others, words, rawCaseAttributes, rawEventAttributes])
        self.word_to_index = dict([(w, i) for i, w in enumerate(self.index_to_word)])
        writeLog("Total number of unique tokens: %d" % len(self.index_to_word))
        if (predict_next_activity):
            writeLog("  # outcomes: %d (predicting next activities)" % len(words))
        else:
            writeLog("  # outcomes: %d" % len(self.outcomes))
        writeLog("  # durations: %d" % len(durations))
        writeLog("  # eventAttributeClusters: %d" % len(eventAttributeClusters))
        writeLog("  # rawEventAttributes: %d" % len(rawEventAttributes))
        writeLog("  # caseAttributeClusters: %d" % len(caseAttributeClusters))
        writeLog("  # rawCaseAttributes: %d" % len(rawCaseAttributes))
        if (predict_next_activity):
            writeLog("  # activity labels: %d + 1 finish token" % len(words))
        else:
            writeLog("  # activity labels: %d" % len(words))
        writeLog("  # others: %d" % 1)

        self.initializeTokenTypeArrays()
        self.positions_train = None if self.final_trace_only else []
        self.seq_length = self.prepareTokenizedSentences(self.traces_train, tokenized_sentences_train, self.positions_train, False) + 1
        writeLog("Maximum sequence length in the training set is %d tokens." % (self.seq_length))

        tokenized_sentences_test = [nltk.WhitespaceTokenizer().tokenize(trace.sentence) for trace in self.traces_test]
        sl = self.prepareTokenizedSentences(self.traces_test, tokenized_sentences_test, None, self.seq_length, True)
        writeLog("Maximum sequence length in the test set is %d tokens." % (sl))

    def initializeFilteredRawAttributes(self, traces):
        includeRawEventAttributes = not self.parameters["disable_raw_event_attributes"] and (self.event_clustering != None)
        includeRawCaseAttributes = not self.parameters["disable_raw_case_attributes"] and (self.case_clustering != None)
        if ((not includeRawCaseAttributes) and (not includeRawEventAttributes)):
            return

        for trace in traces:
            if (includeRawCaseAttributes):
                self.case_clustering.filterCaseAttributes(self.eventlog, trace)
            if (includeRawEventAttributes):
                self.event_clustering.filterEventAttributes(self.eventlog, trace)


    def initializeTokenTypeArrays(self):
        self.unknown_token_id = self.word_to_index[UNKNOWN_TOKEN]
        self.is_outcome = [w.startswith(OUTCOME_SELECTION_TOKEN_PREFIX) for w in self.index_to_word]
        self.is_duration = [w.startswith(DURATION_TOKEN_PREFIX) for w in self.index_to_word]
        self.is_event_attribute_cluster = [w.startswith(EVENT_ATTRIBUTE_TOKEN_PREFIX) for w in self.index_to_word]
        self.is_case_attribute_cluster = [w.startswith(CASE_ATTRIBUTE_TOKEN_PREFIX) for w in self.index_to_word]
        self.is_event_attribute = [(w.startswith(ATTRIBUTE_COLUMN_PREFIX + "E")) for w in self.index_to_word]
        self.is_case_attribute = [(w.startswith(ATTRIBUTE_COLUMN_PREFIX + "C")) for w in self.index_to_word]
        self.is_word_token = [ not (self.is_outcome[i] or self.is_duration[i] or self.is_event_attribute_cluster[i] or self.is_case_attribute_cluster[i] or self.is_event_attribute[i] or self.is_case_attribute[i] or i == self.unknown_token_id) for i, w in enumerate(self.index_to_word)]
        self.num_outcomes = len(self.outcomes)

    def prepareTokenizedSentences(self, traces, tokenized_sentences, positions, initializeFilteredRawAttributes, truncate_to_length = None):
        tokenized_sentences = np.asarray(tokenized_sentences)
        result = self.handleUnknowns(tokenized_sentences, truncate_to_length)
        for i, trace in enumerate(traces):
            trace.tokenized_sentences = tokenized_sentences[i]
        if (positions != None):
            for t, ts in enumerate(tokenized_sentences):
                l = len(ts)
                if l > 1:
                    for pos in range(l - 1):
                        positions.append([t, pos])
        if initializeFilteredRawAttributes:
            self.initializeFilteredRawAttributes(traces)
        return result


    def handleUnknowns(self, tokenized_sentences, truncate_to_length = None):
        # Replace all words not in our vocabulary with the unknown token
        seq_length = 0
        for i, sent in enumerate(tokenized_sentences):
            ts = [w if w in self.word_to_index else UNKNOWN_TOKEN for w in sent]
            if (self.truncate_unknowns):
                origts = ts
                ts = []
                wasUnknown = False
                for w in origts:
                    isUnknown = w == UNKNOWN_TOKEN
                    if ((not isUnknown) or (not wasUnknown)):
                        ts.append(w)
                    wasUnknown = isUnknown
            l = len(ts)
            if (truncate_to_length != None):
                if ts[-1].startswith(OUTCOME_SELECTION_TOKEN_PREFIX):
                    ts = ts[:-1] # Cut the outcome away from the test set if present
                    --l
                if (l > truncate_to_length):
                   ts = ts[:(truncate_to_length)]
            tokenized_sentences[i] = ts
            if (l > seq_length):
                seq_length = l
        return seq_length

    def prepareLayers(self):
        writeLog("Preparing " + str(self.num_layers) + " layers for algorithm: " + self.algorithm)

        # First, we build the network, starting with an input layer
        # Recurrent layers expect input of shape
        # (batch size, SEQ_LENGTH, num_features)
        mask_var = T.matrix('mask')

        l_in = lasagne.layers.InputLayer(shape=(None, None, len(self.word_to_index)))
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
        self.l_out = lasagne.layers.DenseLayer(self.l_layers[len(self.l_layers) - 1], num_units=len(self.word_to_index), W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)
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
        self.rnn_train = theano.function([l_in.input_var, target_values, l_mask.input_var], cost, updates=updates, allow_input_downcast=True)
#!        self.train = theano.function([l_in.input_var, target_var, l_mask.input_var], cost, updates=updates, allow_input_downcast=True)
        writeLog("Compiling train cost computing function...")
        self.compute_cost = theano.function([l_in.input_var, target_values, l_mask.input_var], cost, allow_input_downcast=True)
#!        self.compute_cost = theano.function([l_in.input_var, target_var, l_mask.input_var], cost, allow_input_downcast=True)

        # In order to generate text from the network, we need the probability distribution of the next character given
        # the state of the network and the input (a seed).
        # In order to produce the probability distribution of the prediction, we compile a function called probs. 
        writeLog("Compiling propabilities computing function...")
        self.propabilities = theano.function([l_in.input_var, l_mask.input_var],network_output,allow_input_downcast=True)

    def predict_outcome(self, tracesToCalculateFor, tracePercentage):
        batches, masks = self.gen_prediction_data(tracesToCalculateFor, tracePercentage)
        correct = 0
        predictions = []
        probs_out = []
        predict_next_activity = self.parameters["predict_next_activity"]

        for i in range(len(batches)):
            x = batches[i]
            mask = masks[i]
            probs = self.propabilities(x, mask)
            for prob in enumerate(probs):
                if predict_next_activity:
                    outcomeProbs = np.asarray([p if self.is_word_token[k] or self.is_outcome[k] else 0 for k, p in enumerate(prob[1])])
                else:
                    outcomeProbs = np.asarray([p if self.is_outcome[k] else 0 for k, p in enumerate(prob[1])])
                sumProb = 0
                maxProb = 0
                for t, p in enumerate(outcomeProbs):
                    sumProb += p
                    if (p > maxProb):
                        maxProb = p
                        maxIndex = t
                probs_out.append(maxProb / sumProb)
                word = self.index_to_word[maxIndex]
                if predict_next_activity and word.startswith(OUTCOME_SELECTION_TOKEN_PREFIX):
                    word = word[len(OUTCOME_SELECTION_TOKEN_PREFIX):]
                predictions.append(word)
        return predictions, probs_out

    def createModel(self):
        self.initializeTraces()

        self.layer_preparation_start_time = time()
        self.train_initialization_time_used = self.layer_preparation_start_time - self.train_start_time
        self.prepareLayers()
        self.start_time = time()
        self.layer_initialization_time_used = self.start_time - self.layer_preparation_start_time
        self.previous_time = self.start_time
        self.cumul_train_time = 0
        self.cumul_exact_train_time = 0
        self.cumul_test_time = 0
        self.auc = 0
        self.sr_trains = []
        self.sr_tests = []
        self.sr_vals = []
        self.sr_tests_75p = []
        self.sr_tests_50p = []
        self.sr_tests_25p = []
        self.time_used = []
        self.avg_costs = []
        self.time_used_for_test = []
        self.all_cms = []
        self.best_sr_test = 0
        self.best_sr_vals = 0
        self.best_sr_train = 0
        self.best_num_success = 0
        self.best_num_fail = 0
        self.best_params = None

        def calculateSuccessRate(tracesToCalculateFor, tracePercentage, testId):
            max_num_traces_to_test = self.parameters["max_num_traces_to_test"]
            if (len(tracesToCalculateFor) > max_num_traces_to_test):
                writeLog("Number of traces to test %d exceeds the configured maximum %d. Taking random sample of the configured maximum size." % (len(tracesToCalculateFor), max_num_traces_to_test))
                tracesToCalculateFor = np.asarray(tracesToCalculateFor)[np.random.choice(len(tracesToCalculateFor), max_num_traces_to_test, replace=False)]
            predictions, probs = self.predict_outcome(tracesToCalculateFor, tracePercentage)
            numSuccess = 0
            cm = [0, 0, 0, 0]
            exps = []
            trueWord = self.index_to_word[1] if self.num_outcomes == 2 else ""
            predict_next_activity = self.parameters["predict_next_activity"]
            for i in range(len(tracesToCalculateFor)):
                expected = ("" if predict_next_activity else OUTCOME_SELECTION_TOKEN_PREFIX) + tracesToCalculateFor[i].outcome
                actual = predictions[i]
                numSuccess += 1 if expected == actual else 0
                if (self.num_outcomes == 2):
                    bExpected = expected == trueWord
                    bActual = actual == trueWord
                    exps.append(1 if bExpected else 0)
                    cm[0] += 1 if bExpected and bActual else 0
                    cm[1] += 1 if not bExpected and not bActual else 0
                    cm[2] += 1 if not bExpected and bActual else 0
                    cm[3] += 1 if bExpected and not bActual else 0
            self.cms[testId] = cm
            self.cms_str += ":%i_%i_%i_%i" % (cm[0], cm[1], cm[2], cm[3])
            if ((testId == 1) and (not predict_next_activity) and (self.num_outcomes == 2)):
                self.auc = metrics.roc_auc_score(exps, probs)
            return numSuccess, len(tracesToCalculateFor), numSuccess / len(tracesToCalculateFor)

        def report(num_examples_seen, it, avg_cost, num_report_iterations, test_partial_traces = False):
            t2 = time()
            tutrain = (t2 - self.previous_time)
            self.cumul_train_time = self.cumul_train_time + tutrain
            self.time_used.append(tutrain)
            self.cms = {}
            self.cms_str = ""

            writeLog("Testing 100% training samples")
            numSuccess, numFail, sr_train = calculateSuccessRate(self.traces_train, 1.0, 0)
            self.sr_trains.append(sr_train)
            sr_tests_75p = sr_tests_50p = sr_tests_25p = None

            writeLog("Testing 100% validation samples")
            numSuccess, numFail, sr_vals = calculateSuccessRate(self.traces_validation, 1.0, 2)
            self.cumul_exact_train_time = self.cumul_exact_train_time + (time() - self.previous_time)
            self.sr_vals.append(sr_vals)

            writeLog("Testing 100% test samples")
            numSuccess, numFail, sr_test = calculateSuccessRate(self.traces_test, 1.0, 1)
            self.cumul_exact_train_time = self.cumul_exact_train_time + (time() - self.previous_time)
            self.sr_tests.append(sr_test)

            if (test_partial_traces):
                writeLog("Testing 75% test samples")
                numSuccess, numFail, sr_tests_75p = calculateSuccessRate(self.traces_test, 0.75, 2)
                self.sr_tests_75p.append(sr_tests_75p)
                writeLog("Testing 50% test samples")
                numSuccess, numFail, sr_tests_50p = calculateSuccessRate(self.traces_test, 0.5, 3)
                self.sr_tests_50p.append(sr_tests_50p)
                writeLog("Testing 25% test samples")
                numSuccess, numFail, sr_tests_25p = calculateSuccessRate(self.traces_test, 0.25, 4)
                self.sr_tests_25p.append(sr_tests_25p)
            self.avg_costs.append(avg_cost)
            data_size = len(self.traces_train)
            self.epoch = it*self.batch_size/data_size
            self.generate_trace(5)
            t3 = time()
            tutest = (t3 - t2)
            self.cumul_test_time = self.cumul_test_time + tutest
            self.previous_time = t3
            self.time_used_for_test.append(tutest)
            self.all_cms.append(self.cms)
            writeLog("Iteration: %i (%i) Total time used: ~%f seconds (train: %f, test: %f)" % (num_report_iterations, num_examples_seen, (time() - self.start_time) * 1., self.cumul_train_time, self.cumul_test_time))
            writeLog("Epoch {} average loss = {}".format(self.epoch, avg_cost))
            if (test_partial_traces):
                writeLog("Success rates: test: %f test 75%%: %f test 50%%: %f test 25%%: %f train: %f" % (sr_test, sr_tests_75p, sr_tests_50p, sr_tests_25p, sr_train))
            else:
                writeLog("Success rates: validation: %f test: %f train: %f" % (sr_vals, sr_test, sr_train))

            # IMPORTANT: store validation values
            if (sr_vals > self.best_sr_vals):
                writeLog("Best accuracy thus far achieved. Storing parameters...")
                self.best_sr_test = sr_test
                self.best_sr_vals = sr_vals
                self.best_sr_train = sr_train
                self.best_num_success = numSuccess
                self.best_num_fail = numFail
                self.best_params = lasagne.layers.get_all_param_values(self.l_out, trainable=True)
                self.best_iteration = num_report_iterations

            writeResultRow([datetime.now().replace(microsecond=0).isoformat(), 
                "ok", self.parameters["test_name"], self.case_name, 
                self.parameters["dataset_name"] if (("dataset_name" in self.parameters) and (self.parameters["dataset_name"] != None)) else self.eventlog.filename, 
                self.parameters["cross-validation-run"] if (("cross-validation-run" in self.parameters) and (self.parameters["cross-validation-run"] != None)) else "",
                len(self.traces_train), len(self.traces_test),
                len(self.traces_train) + len(self.traces_test), 
                self.algorithm, self.num_layers, self.hidden_dim_size, 
                self.optimizer, self.learning_rate, self.seq_length, len(self.word_to_index), self.batch_size,
                self.grad_clipping, self.num_iterations_between_reports,
                self.best_iteration,
                num_report_iterations,
                num_examples_seen, self.epoch, 
                self.train_initialization_time_used, self.layer_initialization_time_used,
                self.cumul_exact_train_time, tutrain, self.cumul_train_time, tutest, 
                self.cumul_test_time, sr_train, sr_test, sr_tests_75p, sr_tests_50p,
                sr_tests_25p,
                avg_cost, self.auc, self.cms[1][0], self.cms[1][1], self.cms[1][2], self.cms[1][3],
                str(self.cms_str),
                self.predict_only_outcome, self.final_trace_only, self.trace_length_modifier, 
                self.num_iterations_between_reports * self.num_callbacks == 100000 * 50, 
                self.max_num_words, self.truncate_unknowns, 
                not self.parameters["disable_activity_labels"],
                not self.parameters["disable_durations"],
                not self.parameters["disable_event_attributes"],
                not self.parameters["disable_case_attributes"],
                not self.parameters["disable_raw_event_attributes"],
                not self.parameters["disable_raw_case_attributes"],
                self.parameters["predict_next_activity"],
                self.parameters["use_single_event_clustering"],
                self.parameters["duration_split_method"],
                self.parameters["case_clustering_method"],
                self.parameters["event_clustering_method"],
                self.parameters["case_clustering_include_activity_occurrences"],
                self.parameters["case_clustering_include_case_attributes"],
                self.parameters["include_activity_occurrences_as_raw_case_attributes"],
                self.parameters["use_single_value_for_duration"],
                self.parameters["max_num_case_clusters"],
                self.parameters["max_num_event_clusters"],
                self.parameters["ignore_values_threshold_for_case_attributes"],
                self.parameters["ignore_values_threshold_for_event_attributes"]
            ])
#            self.draw_chart()
    
#        writeLog("Calculating initial probabilities.")
        self.cms = {}
        self.cms_str = ""
#        sr_train = calculateSuccessRate(self.traces_train, 1.0, 0)
#        self.sr_trains.append(sr_train)
#        sr_test = calculateSuccessRate(self.traces_test, 1.0, 1)
#        self.sr_tests.append(sr_test)
#        self.time_used.append(time.time() - self.start_time)
#        self.avg_costs.append(0)
#        writeLog("Initial success rates: test: %f  train: %f" % (sr_test, sr_train))
    
        self.trainModel(report)
    
        self.cms = {}
        self.cms_str = ""
        numSuccess, numFail, sr_train = calculateSuccessRate(self.traces_train, 1.0, 0)
        self.sr_trains.append(sr_train)
        numSuccess, numFail, sr_test = calculateSuccessRate(self.traces_test, 1.0, 1)
        self.sr_tests.append(sr_test)
        self.avg_costs.append(0)
        writeLog("Final success rates: test: %f  train: %f iteration: %i" % (self.best_sr_test, self.best_sr_train, self.best_iteration))
        self.time_used.append(self.cumul_train_time)
        return self.best_num_success, self.best_num_fail, self.cumul_exact_train_time
#        self.draw_chart()

    def generate_trace(self, min_length=2):
        # We start the sentence with the start token
        x = np.zeros((1, self.seq_length, len(self.word_to_index)))
        mask = np.zeros((1, self.seq_length))
        new_sentence = []
        i = 0
        wordsMask = np.array([(b or self.is_outcome[i]) for i, b in enumerate(self.is_word_token)])
        # Repeat until we get an end token
        while ((len(new_sentence) == 0) or (not self.is_outcome[new_sentence[-1]])):
            probs = self.propabilities(x, mask)[0]
            probs = np.asarray([p if wordsMask[i] else 0 for i, p in enumerate(probs)])
            if (probs.sum() == 0.0):
                writeLog("Sum of probabilities is zero. Unable to generate trace.") 
                break
            probs /= probs.sum()
#            samples = np.random.multinomial(1, probs)
#            index = np.argmax(samples)
            index = np.random.choice(range(len(probs)), p=probs)
            new_sentence.append(index)
            x[0, i, index] = 1
            mask[0, i] = 1
            i += 1

            # Seomtimes we get stuck if the sentence becomes too long, e.g. "........" :(
            # And: We don't want sentences with UNKNOWN_TOKEN's
            if len(new_sentence) >= self.seq_length or index == self.word_to_index[UNKNOWN_TOKEN]:
                writeLog("Generated exceedingly long example trace. Skipping.") 
                return None
        if len(new_sentence) < min_length:
            return None
        res = [self.index_to_word[x] for x in new_sentence]
        writeLog("Generated example trace of length %d: %s" % (len(res), str(res))) 
        return res

    def save(self):
        saved = {
            "name": self.eventlog.filename,
            "nn_params": {
                "algorithm": self.algorithm,
                "num_layers": self.num_layers,
                "optimizer": self.optimizer,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_callbacks": self.num_callbacks,
                "case_name": self.case_name,
                "hidden_dim_size": self.hidden_dim_size,
                "num_iterations_between_reports": self.num_iterations_between_reports,
                "grad_clipping": self.grad_clipping,
                "predict_only_outcome": self.predict_only_outcome,
                "final_trace_only": self.final_trace_only,
                "max_num_words": self.max_num_words,
                "trace_length_modifier": self.trace_length_modifier,
                "truncate_unknowns": self.truncate_unknowns,
                "trained_params": self.best_params
            },
            "event_clustering": self.event_clustering,
            "case_clustering": self.case_clustering,
            "eventlog": {
                "activities": self.eventlog.data["activities"],
                "attributes": self.eventlog.data["attributes"],
                "filename": self.eventlog.filename,
                "filepath": self.eventlog.filepath
            },
            "word_to_index": self.word_to_index,
            "index_to_word": self.index_to_word,
            "seq_length": self.seq_length,
            "outcomes": self.outcomes,
            "parameters": self.parameters
        }
        return saved

    def load(self, saved):
        self.algorithm = saved["nn_params"]["algorithm"]
        self.num_layers = saved["nn_params"]["num_layers"]
        self.optimizer = saved["nn_params"]["optimizer"]
        self.learning_rate = saved["nn_params"]["learning_rate"]
        self.batch_size = saved["nn_params"]["batch_size"]
        self.num_callbacks = saved["nn_params"]["num_callbacks"]
        self.case_name = saved["nn_params"]["case_name"]
        self.hidden_dim_size = saved["nn_params"]["hidden_dim_size"]
        self.num_iterations_between_reports = saved["nn_params"]["num_iterations_between_reports"]
        self.grad_clipping = saved["nn_params"]["grad_clipping"]
        self.predict_only_outcome = saved["nn_params"]["predict_only_outcome"]
        self.final_trace_only = saved["nn_params"]["final_trace_only"]
        self.max_num_words = saved["nn_params"]["max_num_words"]
        self.trace_length_modifier = saved["nn_params"]["trace_length_modifier"]
        self.truncate_unknowns = saved["nn_params"]["truncate_unknowns"]
        self.word_to_index = saved["word_to_index"]
        self.index_to_word = saved["index_to_word"]
        self.seq_length = saved["seq_length"]
        self.outcomes = saved["outcomes"]
        self.parameters.update(saved["parameters"])
        try:
            self.event_clustering = saved["event_clustering"]
            self.event_clustering.addUndefinedParameters(self.parameters)
            self.case_clustering = saved["case_clustering"]
            self.case_clustering.addUndefinedParameters(self.parameters)
            self.eventlogActivities = saved["eventlog"]["activities"]
            self.eventlogAttributes = saved["eventlog"]["attributes"]
            self.eventlogFilename = saved["eventlog"]["filename"]
            self.eventlogFilepath = saved["eventlog"]["filepath"]
            self.prepareLayers()
            lasagne.layers.set_all_param_values(self.l_out, saved["nn_params"]["trained_params"], trainable=True)
        except:
            writeLog("Exception: " + sys.exc_info()[0])

    def test(self, eventlog, tracePercentage = 1.0, maxNumTraces = None):
        self.eventlog = eventlog
        self.eventlog.model = self
        eventlog.initializeForTesting(self)

        self.prepareTestData(eventlog)
        self.traces_train = []
        self.traces_test = eventlog.convertTracesFromInputData(eventlog.testData, self.parameters, self.trace_length_modifier)
        if (maxNumTraces != None) and (maxNumTraces < len(self.traces_test)):
            writeLog("Filtering %d traces out of %d test traces" % (maxNumTraces, len(self.traces_test)))
            self.traces_test = list(np.random.choice(np.asarray(self.traces_test), maxNumTraces, replace=False))

        self.initializeTokenTypeArrays()

        tokenized_sentences_test = [nltk.WhitespaceTokenizer().tokenize(trace.sentence) for trace in self.traces_test]
        sl = self.prepareTokenizedSentences(self.traces_test, tokenized_sentences_test, None, self.seq_length, True)
        writeLog("Maximum sequence length in the test set is %d tokens." % (sl))

        predictions, probs = self.predict_outcome(self.traces_test, tracePercentage)
        numSuccess = 0
        cases = eventlog.data["cases"]
        if len(cases) > 0:
            predict_next_activity = self.parameters["predict_next_activity"]

            if (predict_next_activity or ("s" in cases[0])):
                prefix = ("" if predict_next_activity else OUTCOME_SELECTION_TOKEN_PREFIX)
                for i, pred in enumerate(predictions):
                    if pred == prefix + self.traces_test[i].outcome:
                        numSuccess += 1

        print("numSucess: ", numSuccess)
        return self.traces_test, predictions, probs, numSuccess
        
