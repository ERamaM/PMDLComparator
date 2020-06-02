#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 11:59:43 2017

Test framework sources used to perform the tests required by paper: 
"hinkka"
by Markku Hinkka, Teemu Lehto and Keijo Heljanko
"""
import sys
import numpy as np
import json
import logging
from time import time
from optparse import OptionParser
import collections
import datetime
import copy

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import KFold
from sklearn import metrics
from pathlib import Path
import ntpath

from model import Model, DURATION_TOKEN_PREFIX, EVENT_ATTRIBUTE_TOKEN_PREFIX, WORD_PART_SEPARATOR
from my_utils import writeLog, generate_traces, configure, TraceData, getInputDatasetFilename
from modelcluster import ModelCluster
from cluster import Clustering
from pathlib import Path
import pandas as pd
import os

def parse_date(jd):
    sign = jd[-7]
    if sign not in '-+' or len(jd) == 13:
        millisecs = int(jd[6:-2])
    else:
        millisecs = int(jd[6:-7])
        hh = int(jd[-7:-4])
        mm = int(jd[-4:-2])
        if sign == '-': mm = -mm
        millisecs += (hh * 60 + mm) * 60000
    return datetime.datetime(1970, 1, 1) \
        + datetime.timedelta(microseconds=millisecs * 1000)

class EventLog:
    def __init__(self, parameters, rng, filename = None, pTraining = 0.0, modelCluster = None, inputJson = None):
        writeLog("Initializing event log")

        self.rng = rng
        self.parameters = dict(parameters)
        self.trainingData = []
        self.validationData = []
        self.testData = []

        # Filename is not null and inputJson is always null
        # Open the csvs and count the number of events
        print("FILENAME EVENTLOG: ", filename)
        print("FILENAME INPUTJSON: ", inputJson)
        print("PARAMETERS: ", parameters)
        if filename is not None:
            name = Path(filename).stem
            train_df = pd.read_csv(os.path.join("testdata", "train_" + name + ".csv"))
            val_df = pd.read_csv(os.path.join("testdata", "val_" + name + ".csv"))
            test_df = pd.read_csv(os.path.join("testdata", "test_" + name + ".csv"))
            self.train_cases = len(train_df.groupby("case:concept:name"))
            self.val_cases = len(val_df.groupby("case:concept:name"))
            self.test_cases = len(test_df.groupby("case:concept:name"))
            print("Train evn: ", self.train_cases)
            print("Val evn: ", self.val_cases)
            print("Test evn: ", self.test_cases)


        if (inputJson != None):
            self.data = json.loads(inputJson)
            self.filename = "unnamed"
            self.filepath = ""
        elif (filename != None):
            path = Path(filename)
            if (not path.is_file()):
                filename = getInputDatasetFilename(filename)
            self.filepath = filename
            self.filename = ntpath.basename(filename)
            with open(filename) as f:
                self.data = json.load(f)
        else:
            return

        self.pTraining = pTraining
        if pTraining == None:
            return

        if (modelCluster != None):
            model = modelCluster.models[0]
            if not ("activities" in self.data):
                self.data["activities"] = model.eventlogActivities
            if not ("attributes" in self.data):
                self.data["attributes"] = model.eventlogAttributes
        self.setTrainingSize(parameters, pTraining)
        self.initializationReport()

    def initializationReport(self):
        writeLog("Initialized event log %s" % (self.filename))
        writeLog("  # cases: %d (train: %d, test: %d)" % (len(self.data["cases"]), len(self.trainingData), len(self.testData)))
        writeLog("  # activities: %d" % (len(self.data["activities"])))
        writeLog("  # case attributes: %d" % (len(self.data["attributes"]["case"])))
        writeLog("  # event attributes: %d" % (len(self.data["attributes"]["event"])))
        if (self.pTraining != None):
            writeLog("  Training set percentage: %d" % (int(self.pTraining * 100)))

    def initializeForTesting(self, model):
        self.trainingData = []
        self.testData = np.asarray(self.data["cases"])
        if (model.eventlogActivities != None):
            self.data["activities"] = model.eventlogActivities
        if (model.eventlogAttributes != None):
            self.data["attributes"] = model.eventlogAttributes
        if (model.eventlogFilename != None):
            self.filename = model.eventlogFilename
        if (model.eventlogFilepath != None):
            self.filepath = model.eventlogFilepath
        self.initializeDerivedData()

    def createEmptyCopy(self, parameters = None):
        result = EventLog(parameters if parameters != None else self.parameters, self.rng)
        result.trainingData = []
        result.testData = []
        result.filename = self.filename
        result.filepath = self.filepath
        result.data = {
            "cases": [],
            "activities": copy.copy(self.data["activities"]) if ("activities" in self.data) else None,
            "attributes": copy.copy(self.data["attributes"] if ("attributes" in self.data) else None)
        }
        result.parent = self
        return result

    def setTrainingSize(self, parameters, pTraining):
        cases = np.asarray(self.data["cases"])

        maxNumCases = parameters["max_num_cases_in_training"]
        if (maxNumCases != None) and (maxNumCases < len(cases)):
            writeLog("Filtering out %d cases out of %d" % (maxNumCases, len(cases)))
            cases = np.random.choice(cases, maxNumCases, replace=False)
            self.data["cases"] = cases

        # Load the splits exactly from the files
        self.trainingData = cases[:self.train_cases]
        self.validationData = cases[self.train_cases:self.train_cases + self.val_cases]
        self.testData = cases[self.train_cases + self.val_cases:]
        self.initializeDerivedData()

    def getActivityOccurrences(self, cases):
        result = copy.copy(self.activities)
        for activityId, activity in result.items():
            activity["occ"] = []
        for c in cases:
            for e in c["t"]:
                activity = result[e[0]]
                activity["occ"].append(e)
        return result

    def initializeDerivedData(self, forSplittedEventLog = False):        
        self.activities = {}
        self.activitiesByLabel = {}
        if ("activities" in self.data):
            for a in self.data["activities"]:
                self.activities[a["id"]] = {
                    "name": a["name"],
                    "occ": []
                }
                self.activitiesByLabel[a["name"].replace(" ", "_")] = a

            if (not forSplittedEventLog):
                writeLog("Initializing activity counts for %d cases" % (len(self.data["cases"])))
                for c in self.data["cases"]:
                    counters = collections.Counter(t[0] for t in c["t"])
                    c["occ"] = [counters[act["id"]] for act in self.data["activities"]]

        self.flows = {}

    def preProcessForTraining(self, parameters):
        disableDurations = parameters["disable_durations"]
        if not disableDurations:
            numEvents = 0
            writeLog("Pre-processing %d cases" % (len(self.trainingData)))
            for c in self.trainingData:
                prev = None
                prevDate = None
                evts = c["t"]
                numEvents += len(evts)
                for e in evts:
                    eDate = parse_date(e[1])
                    if prev is not None:
                        key = "%s->%s" % (prev[0], e[0])
                        if (key in self.flows):
                            flow = self.flows[key]
                        else:
                            flow = self.flows[key] = { "name": key, "occ": [] }
                        delta = eDate - prevDate
                        flow["occ"].append(delta)
                    prevDate = eDate
                    prev = e

            writeLog("Total number of events in training data: %d (Average case length: %f)" % (numEvents, (numEvents / len(self.trainingData))))
            writeLog("Pre-processing %d flows" % (len(self.flows)))
            for key in self.flows:
                f = self.flows[key]
                nOcc = len(f["occ"])
                f["occ"].sort()
                if (nOcc > 0):
                    min = f["min"] = f["occ"][0]
                    max = f["max"] = f["occ"][nOcc - 1]
                    f["avg"] = np.mean(f["occ"])
                    f["med"] = np.median(f["occ"])
                    f["perc10"] = np.percentile(f["occ"], 10)
                    f["perc25"] = np.percentile(f["occ"], 25)
                    f["perc75"] = np.percentile(f["occ"], 75)
                    f["perc90"] = np.percentile(f["occ"], 90)
                    f["diff"] = f["max"] - f["min"]
                    f["fast"] = f["perc10"]
                    f["slow"] = f["perc90"]
        
    def addTrace(self, traceData, isForTraining):
        self.data["cases"].append(traceData)
        if (isForTraining):
            self.trainingData.append(traceData)
        else:
            self.testData.append(traceData)

    def convertTracesFromInputData(self, data, parameters, trace_length_modifier):
        writeLog("Converting %d cases into event traces." % (len(data)))

        enableDurations = not parameters["disable_durations"]
        splitDurationsInto5Buckets = parameters["duration_split_method"] == "5-buckets"
        addOnlyFullTraceForFinisher = not parameters["predict_next_activity"]
        useSingleValueForDuration = parameters["use_single_value_for_duration"]
        includeActivityOccurrencesAsRawCaseAttributes = parameters["include_activity_occurrences_as_raw_case_attributes"]
        disableEventAttributes = parameters["disable_event_attributes"]
        splitTracesToPrefixes = parameters["split_traces_to_prefixes"]
        minPrefixLength = parameters["min_splitted_trace_prefix_length"]
        maxTraceLength = parameters["max_trace_length"]

        result = []
        numFilteredCases = 0
        numFilteredTraces = 0
        for c in data:
            traces = []
            l = len(c["t"])
            finisherTraceFiltered = False

            if l > minPrefixLength:
                if splitTracesToPrefixes:
                    if (l > maxTraceLength):
                        numFilteredCases += 1
                        numFilteredTraces += l - maxTraceLength - minPrefixLength
                        l = maxTraceLength
                        finisherTraceFiltered = True
                    for i in range(minPrefixLength, l):
                        traces.append(c["t"][:i])
                else:
                    if (l > maxTraceLength):
                        numFilteredCases += 1
                        numFilteredTraces += 1
                        finisherTraceFiltered = True
                        traces.append(c["t"][:maxTraceLength])
                    else:
                        traces.append(c["t"])

            if len(traces) == 0:
                continue

            lastTrace = traces[len(traces) - 1]
            for trace in traces:
                sentence = []
                durations = []
                cAttributes = (c["a"] + c["occ"]) if includeActivityOccurrencesAsRawCaseAttributes else c["a"]
                prev = None
                prevDate = None
                eAttributes = []
                for e in trace:
                    eDate = parse_date(e[1])
                    durationPart = DURATION_TOKEN_PREFIX + "normal"
                    dp = 0.5
                    if enableDurations and prev is not None:
                        key = "%s->%s" % (prev[0], e[0])
                        flow = self.flows[key] if key in self.flows else None
                        delta = eDate - prevDate
                        if (flow != None) and ("slow" in flow):
                            if splitDurationsInto5Buckets:
                                if (delta > flow["perc90"]):
                                    durationPart = DURATION_TOKEN_PREFIX + "perc90"
                                    dp = 0.0
                                elif (delta > flow["perc75"]):
                                    durationPart = DURATION_TOKEN_PREFIX + "perc75"
                                    dp = 0.25
                                elif (delta > flow["perc25"]):
                                    durationPart = DURATION_TOKEN_PREFIX + "perc25"
                                    dp = 0.5
                                elif (delta > flow["perc10"]):
                                    durationPart = DURATION_TOKEN_PREFIX + "perc10"
                                    dp = 0.75
                                else:
                                    durationPart = DURATION_TOKEN_PREFIX + "perc0"
                                    dp = 1.0
                            else:
                                if (delta > flow["slow"]):
                                    durationPart = DURATION_TOKEN_PREFIX + "slow"
                                    dp = 0.0
                                elif (delta < flow["fast"]):
                                    durationPart = DURATION_TOKEN_PREFIX + "fast"
                                    dp = 1.0
                    actPart = self.activities[e[0]]["name"]
                    eAttributes += [e[2:(len(e) - 1) if disableEventAttributes else -1]]
                    clusterPart = EVENT_ATTRIBUTE_TOKEN_PREFIX + str(e[len(e) - 1])
                    sentence.append(durationPart + WORD_PART_SEPARATOR + actPart.replace(WORD_PART_SEPARATOR, "_") + WORD_PART_SEPARATOR + clusterPart)
                    if useSingleValueForDuration:
                        durations.append(dp)
                    prevDate = eDate
                    prev = e
                finisher = c["f"] if "f" in c else ((trace == lastTrace) and (not finisherTraceFiltered))
                cluster = c["_cluster"] if ("_cluster" in c) else None
                if (not (addOnlyFullTraceForFinisher and finisher)):
                    result.append(TraceData(c["n"], c["s"] if "s" in c else None, "s" in c, cAttributes, eAttributes, cluster, sentence, durations, parameters, trace_length_modifier, self.model, False))
                if (finisher):
                    result.append(TraceData(c["n"] + "_f", c["s"] if "s" in c else None, "s" in c, cAttributes, eAttributes, cluster, sentence, durations, parameters, trace_length_modifier, self.model, True))
        writeLog("Generated %d event traces out of %d cases." % (len(result), len(data)))
        if (numFilteredTraces > 0):
            writeLog("Filtered %d traces in %d cases due to them having more than maximum allowed number of events (%d)" % (numFilteredTraces, numFilteredCases, maxTraceLength))
        return result

    def performCrossValidationRun(self, fullTestData, trainIndex, testIndex, parameters):
        maxNumCases = parameters["max_num_cases_in_training"]
        cvRunIndex = parameters["cross-validation-run"]
        nSplits = parameters["cross-validation-splits"]

        writeLog("Starting cross-validation run %d of %d" % (cvRunIndex, nSplits))

        if (maxNumCases != None) and (maxNumCases < len(trainIndex)):
            writeLog("Filtering out %d cases out of %d" % (maxNumCases, len(trainIndex)))
            trainIndex = np.random.choice(trainIndex, maxNumCases, replace=False)

        runEventLog = self.createEmptyCopy()

        runEventLog.data["cases"] = fullTestData[trainIndex]
        runEventLog.pTraining = parameters["test_data_percentage"]
        runEventLog.setTrainingSize(parameters, runEventLog.pTraining)
        runEventLog.initializationReport()

        m = ModelCluster(runEventLog.rng)
        m.initialize(
                parameters = parameters,
                case_clustering = Clustering(parameters["case_clustering_method"], parameters, { 
                    "num_clusters": parameters["num_case_clusters"],
                    "max_num_clusters": parameters["max_num_case_clusters"],
                    "ignore_values_threshold": parameters["ignore_values_threshold_for_case_attributes"]
                }),
                event_clustering = Clustering(parameters["event_clustering_method"], parameters, { 
                    "num_clusters": parameters["num_event_clusters"],
                    "max_num_clusters": parameters["max_num_event_clusters"],
                    "ignore_values_threshold": parameters["ignore_values_threshold_for_event_attributes"]
                }),
                rng = runEventLog.rng)
        trainResult = m.train(runEventLog)

        writeLog("Starting cross-validation test for run %d" % (cvRunIndex))

        runEventLog = self.createEmptyCopy()
        runEventLog.data["cases"] = fullTestData[testIndex]
        runEventLog.testData = fullTestData[testIndex]
        runEventLog.trainingData = []
        runEventLog.pTraining = 0.0
        runEventLog.initializeDerivedData()
        runEventLog.initializationReport()
        maxNumTraces = parameters["max_num_traces_in_testing"] if "max_num_traces_in_testing" in parameters else None
        m.test(runEventLog, 1.0, trainResult, maxNumTraces)

    @staticmethod
    def performCrossValidatedTests(parameters, inputJson, rng):
        e = EventLog(parameters, rng,  parameters["input_filename"], None, inputJson = inputJson)
        e.performCrossValidatedTestsForFullEventLog()

    def performCrossValidatedTestsForFullEventLog(self):
        parameters = self.parameters
        nSplits = parameters["cross-validation-splits"]
        writeLog("Performing cross-validation using %d splits" % (nSplits))

        fullTestData = np.asarray(self.data["cases"])

        self.initializationReport()

        # TODO: parte conflictiva.
        # DESACTIVAR EL SHUFFLING. nSplits siempre tiene que ser 2 (train test)
        kf = KFold(n_splits=nSplits, random_state=self.rng, shuffle=False)
        cvRunIndex = 0
        for trainIndex, testIndex in kf.split(fullTestData):
            cvRunIndex += 1
            parameters["cross-validation-run"] = cvRunIndex
            self.performCrossValidationRun(fullTestData, trainIndex, testIndex, parameters)
