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
#import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
from time import time
from pathlib import Path
from cluster import Clustering
import pandas as pd
import nltk
import itertools
import json
from my_utils import TraceData, writeLog, writeResultRow, writeTestResultRow, get_filename, getOutputPath, OUTCOME_SELECTION_TOKEN_PREFIX, DURATION_TOKEN_PREFIX, EVENT_ATTRIBUTE_TOKEN_PREFIX, WORD_PART_SEPARATOR, CASE_ATTRIBUTE_TOKEN_PREFIX
from bucket import Bucket
from model import Model

UNKNOWN_TOKEN = "UNKNOWN"

class ModelCluster:
    def __init__(self, rng):
        lasagne.random.set_rng(rng)
        writeLog("Creating new model cluster object")

    def initialize(self, parameters,
                case_clustering, event_clustering, rng):
        self.caseClusterModel = None
        self.caseClusterVectorizer = None

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
        self.num_models = parameters["num_models"]
        self.parameters = parameters
        self.case_clustering = case_clustering
        self.event_clustering = event_clustering
        self.rng = rng

        self.models = [self.createModel() for i in range(self.num_models)]

    def createModel(self):
        result = Model(self.parameters)
        result.initialize(
                case_clustering = Clustering(copyFrom=self.case_clustering),
                event_clustering = Clustering(copyFrom=self.event_clustering),
                rng = self.rng)
        return result

    def train(self, eventlog):
        self.eventlogs = self.splitLog(eventlog)

        writeLog("Trace distribution by models:")
        trainDatasetSize = 0
        for i, eventlog in enumerate(self.eventlogs):
            writeLog("Model #%d: Train: %d traces, Test: %d traces" % (i + 1, len(eventlog.trainingData), len(eventlog.testData)))
            trainDatasetSize += len(eventlog.trainingData) + len(eventlog.testData)

        tutrain = 0
        numSuccess = 0
        numFail = 0
        titu = 0
        litu = 0
        numEpochs = []
        ivs = []
        bestIterations = []
        for i, eventlog in enumerate(self.eventlogs):
            model = self.models[i]
            writeLog("Training model %d of %d" % (i + 1, len(self.eventlogs)))
            ns, ne, tu = model.train(eventlog)
            numEpochs.append(model.epoch)
            ivs.append(len(model.word_to_index))
            bestIterations.append(model.best_iteration)
            tutrain += tu
            numSuccess += ns
            numFail += ne
            titu += model.train_initialization_time_used
            litu += model.layer_initialization_time_used
        srtrain = numSuccess / numFail
        writeLog("Total time used in training: %d (success rate = %f)" % (tutrain, srtrain))
        return { 
            "success_rate": srtrain,
            "train_dataset_size": trainDatasetSize,
            "train_time_used": tutrain,
            "train_init_time_used": titu, 
            "layer_init_time_used": litu,
            "num_epochs": np.mean(np.asarray(numEpochs)),
            "test_iterations": self.parameters["num_callbacks"],
            "input_vector_size": np.mean(ivs),
            "best_iteration": np.mean(bestIterations)
        }

    def splitLog(self, eventlog, onlyTest = False):
        self.eventlog = eventlog
        true_k = len(self.models)
        if (true_k == 1):
            return [self.eventlog]

        t0 = time()
        result = [self.eventlog.createEmptyCopy(self.parameters) for model in self.models]

        if (not onlyTest):
            cases = np.array([c["occ"] for c in self.eventlog.trainingData])
            df = pd.DataFrame(cases, columns=[a["name"] for a in self.eventlog.data["activities"]])

            self.caseClusterVectorizer = DictVectorizer(sparse = False)
            X = self.caseClusterVectorizer.fit_transform(df.to_dict(orient = 'records'))

            writeLog("Event log splitting done in %fs" % (time() - t0))
            writeLog("n_samples: %d, n_features: %d" % X.shape)

            # #############################################################################
            # Do the actual clustering
    #        if opts.minibatch:
            self.caseClusterModel = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                 init_size=1000, batch_size=1000, verbose=False)
    #        else:
    #            self.caseClusterModel = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
    #                        verbose=opts.verbose)

    #        writeLog("Clustering sparse data with %s" % self.caseClusterModel)
            t0 = time()
            x = self.caseClusterModel.fit(X)
            writeLog("done in %0.3fs" % (time() - t0))

            for i, d in enumerate(x.labels_):
                result[d].addTrace(self.eventlog.trainingData[i], True)

        cases = np.array([c["occ"] for c in self.eventlog.testData])
        df = pd.DataFrame(cases, columns=[a["name"] for a in self.eventlog.data["activities"]])
        XX = self.caseClusterVectorizer.transform(df.to_dict(orient = 'records'))
        x = self.caseClusterModel.predict(XX)
        for i, d in enumerate(x):
            result[d].addTrace(self.eventlog.testData[i], False)

        for eventlog in result:
            eventlog.initializeDerivedData(True)

        return result

    def save(self, file_handle, parameters):
        hasExactOutputFilename = (("output_filename" in parameters) and (parameters["output_filename"] != None))
        directory = parameters["model_output_directory"] if ("model_output_directory" in parameters and parameters["model_output_directory"] != None) else getOutputPath()
        filename = parameters["output_filename"] if hasExactOutputFilename else (((parameters["dataset_name"] + "-") if (("dataset_name" in parameters) and (parameters["dataset_name"] != None)) else "") + parameters["test_name"])
        filename = (directory + filename) if hasExactOutputFilename else ("%s%s_%s.model" % (directory, file_handle, filename))
        savedModels = [model.save() for model in self.models]
        saved = {
            "name": self.eventlog.filename,
            "parameters": self.parameters,
            "saved_models": savedModels,
            "case_cluster_model": self.caseClusterModel,
            "case_cluster_vectorizer": self.caseClusterVectorizer,
            "case_clustering": self.case_clustering,
            "event_clustering": self.event_clustering,
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
                "num_models": self.num_models
            }
        }
        with open(filename, 'wb') as f:
            pickle.dump(saved, f) # https://groups.google.com/d/msg/lasagne-users/w8safJOJYvI/SvdiuIHIDQAJ
        return filename

    def load(self, filename, parameters):
        path = Path(filename)
        if (not path.is_file()):
            filename = getOutputPath() + filename
        with open(filename, 'rb') as f:
            saved = pickle.load(f) # https://groups.google.com/d/msg/lasagne-users/w8safJOJYvI/SvdiuIHIDQAJ
        self.parameters = dict(parameters)
        self.parameters.update(saved["parameters"])
        self.caseClusterModel = saved["case_cluster_model"]
        self.caseClusterVectorizer = saved["case_cluster_vectorizer"]
        self.case_clustering = saved["case_clustering"]
        self.event_clustering = saved["event_clustering"]
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
        self.num_models = saved["nn_params"]["num_models"]
        self.models = []
        for i in range(self.num_models):
            writeLog("Loading model %d of %d" % (i + 1, self.num_models))
            model = Model(self.parameters)
            self.models.append(model)
            model.load(saved["saved_models"][i])

    def test(self, eventlog, tracePercentage = 1.0, trainResult = None, maxNumTraces = None):
        self.eventlogs = self.splitLog(eventlog, True)

        writeLog("Trace distribution by models:")
        for i, eventlog in enumerate(self.eventlogs):
            writeLog("Model #%d: Train: %d cases, Test: %d cases" % (i + 1, len(eventlog.trainingData), len(eventlog.testData)))

        traces = []
        predictions = []
        probs = []
        numSuccess = 0

        t0 = time()
        for i, model in enumerate(self.models):
            writeLog("Testing model %d of %d" % (i + 1, len(self.eventlogs)))
            t, pred, prob, ns, real_probs, index_predictions, ground_truth = model.test(self.eventlogs[i], tracePercentage, maxNumTraces, fullProbs=True)
            traces += t
            predictions += pred
            probs += prob
            numSuccess += ns

        tutest = (time() - t0)
        sr_test = numSuccess / len(predictions)

        import os
        print("Model cluster parameters: ", self.parameters)
        filename = self.parameters["test_filename"] if self.parameters["test_filename"] is not None else self.parameters["dataset_name"]
        with open(os.path.join("output", "results_" + filename), "w") as result_file:
            result_file.write("Accuracy: " + str(sr_test))
            result_file.write("Len preds: " + str(len(index_predictions)))
            result_file.write("Len gt: " + str(len(ground_truth)))
            # TODO: zumbarle ahí el resto de metricas a ver si se puede

        writeLog("Success rate for test data: %d/%d (=%f%%)" % (numSuccess, len(predictions), 100 * sr_test))

        train_success_rate = ""
        train_time_used = ""
        train_init_time_used = ""
        train_layer_init_time_used = ""
        num_epochs = ""
        test_iterations = ""
        train_dataset_size = 0
        if trainResult != None:
            train_success_rate = trainResult["success_rate"]
            train_time_used = trainResult["train_time_used"]
            train_init_time_used = trainResult["train_init_time_used"]
            train_layer_init_time_used = trainResult["layer_init_time_used"]
            train_dataset_size = trainResult["train_dataset_size"]
            num_epochs = trainResult["num_epochs"]
            test_iterations = trainResult["test_iterations"]
            train_input_vector_size = trainResult["input_vector_size"]
            train_best_iteration = trainResult["best_iteration"]
        """
        writeTestResultRow([datetime.now().replace(microsecond=0).isoformat(), 
            "ok-test", self.parameters["test_name"], self.case_name, 
            self.parameters["dataset_name"] if (("dataset_name" in self.parameters) and (self.parameters["dataset_name"] != None)) else self.eventlog.filename, 
            self.parameters["cross-validation-run"] if (("cross-validation-run" in self.parameters) and (self.parameters["cross-validation-run"] != None)) else "",
            train_dataset_size, len(traces), len(traces), 
            self.algorithm, self.num_layers, self.hidden_dim_size, 
            self.optimizer, self.learning_rate, "", train_input_vector_size, self.batch_size,
            self.grad_clipping, self.num_iterations_between_reports,
            train_best_iteration,
            test_iterations, "", num_epochs, 
            train_init_time_used, train_layer_init_time_used,
            train_time_used, train_time_used, train_time_used, tutest, tutest, train_success_rate, sr_test, "", "",
            "",
            "", "", "", "", "", "",
            "",
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
        """

        writeLog("Collecting results...")
        result = {}
        for i, trace in enumerate(traces):
            pred = predictions[i]
            result[trace.traceId] = {
                "outcome": pred[len(OUTCOME_SELECTION_TOKEN_PREFIX):] if pred.startswith(OUTCOME_SELECTION_TOKEN_PREFIX) else pred,
                "p": probs[i],
                "expected": trace.outcome if trace.outcome != None else ""
            }
        return result
