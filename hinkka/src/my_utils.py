#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 11:59:43 2017

Test framework sources used to perform the tests required by paper: 
"hinkka"
by Markku Hinkka, Teemu Lehto and Keijo Heljanko
"""

import csv
import numpy as np
import time
import sys
import operator
import io
import array
from datetime import datetime
#import matplotlib.pyplot as plt
import math

OTHER_TOKEN = "OTHER"
UNKNOWN_TOKEN = "UNKNOWN"
ATTRIBUTE_COLUMN_PREFIX = "_A"
OUTCOME_SELECTION_TOKEN_PREFIX = "_O_"
DURATION_TOKEN_PREFIX = "_D_"
EVENT_ATTRIBUTE_TOKEN_PREFIX = "_EA_"
CASE_ATTRIBUTE_TOKEN_PREFIX = "_CA_"
WORD_PART_SEPARATOR = ":"
TRACE_FINISH_TOKEN = "__FINISH__"

class TraceData:
    traceId = ""
    outcome = None
    activities = []
#    sentence = ""
    tokenized_sentences = None
    positions = []
    activityLabels = []
    durations = []
    durationValues = []
    eventAttributeClusters = []

    def __init__(self, traceId, outcome, outcomeDefined, cAttributes, eAttributes, cluster, sentence, durationValues, parameters, trace_length_modifier, model, is_full_trace = False):
        self.traceId = traceId
        self.fullActivities = np.asarray([w.replace(" ", "_") for w in sentence])
        if parameters["predict_next_activity"]:
#            self.fullActivities = np.asarray([w.replace(" ", "_") for w in sentence] + [FINISH_TOKEN])
            if is_full_trace:
                self.outcome = TRACE_FINISH_TOKEN
                self.outcomeToken = OUTCOME_SELECTION_TOKEN_PREFIX + self.outcome
            elif not parameters["predict_only"]:
                last = self.fullActivities[-1]
                parts = last.split(WORD_PART_SEPARATOR)
                w = parts[1] if len(parts) == 3 else last
                self.outcomeToken = self.outcome = w
                self.fullActivities = self.fullActivities[:-1]
                eAttributes = eAttributes[:-1]
        else:
            self.outcome = str(outcome)
            self.outcomeToken = OUTCOME_SELECTION_TOKEN_PREFIX + self.outcome
        self.caseAttributeCluster = CASE_ATTRIBUTE_TOKEN_PREFIX + str(cluster)
        self.trace_length_modifier = trace_length_modifier
        if (trace_length_modifier != 1.0):
            self.activities = self.fullActivities[range(math.ceil(trace_length_modifier * len(self.fullActivities)))]
        else:
            self.activities = self.fullActivities
        self.sentence = "%s %s" % (" ".join(self.activities), OUTCOME_SELECTION_TOKEN_PREFIX + str(self.outcome))
        self.activitiesForPrediction = {}
        self.durations = []
        self.activityLabels = []
        self.eventAttributeClusters = []
        self.durationValues = np.asarray(durationValues)
        for a in self.activities:
            parts = a.split(WORD_PART_SEPARATOR)
            if (len(parts) == 3):
                self.durations.append(parts[0])
                self.activityLabels.append(parts[1])
                self.eventAttributeClusters.append(parts[2])
            else:
                self.durations.append(None)
                self.activityLabels.append(a)
                self.eventAttributeClusters.append(None)

        self.activityLabels = np.asarray(self.activityLabels)
        self.durations = np.asarray(self.durations)
        self.eventAttributeClusters = np.asarray(self.eventAttributeClusters)
        self.eventAttributes = [] if parameters["disable_raw_event_attributes"] else eAttributes
        self.caseAttributes = [] if parameters["disable_raw_case_attributes"] else cAttributes
        self.filteredCaseAttributes = []
        self.filteredEventAttributes = []
        self.indexedCaseAttributeWords = None
        self.indexedEventAttributeWords = None

    def getActivitiesForPrediction(self, word_to_index, tracePercentage, truncateUnknowns, seqLength, vocabSize, disableActivityLabels, disableDurations, disableEventAttributes, disableCaseAttributes, useSingleValueForDuration):
#        key = "%s_%s_%s_%s_%s" % (tracePercentage, self.trace_length_modifier, truncateUnknowns, seqLength, vocabSize) 
#        if (not key in self.activitiesForPrediction):
        r = range(math.ceil(tracePercentage * len(self.activityLabels)))
        labels = self.activityLabels[r]
        durations = self.durations[r]
        eventAttributeClusters = self.eventAttributeClusters[r]
        durationValues = self.durationValues[r] if useSingleValueForDuration else None
#            unknownId = word_to_index[UNKNOWN_TOKEN]
#            wordTokens = [word_to_index[word] if (word in word_to_index) else unknownId for word in sentence]
        labels = [(word_to_index[word] if (word in word_to_index) and (not disableActivityLabels) else word_to_index[UNKNOWN_TOKEN]) for word in labels]
        durations = [word_to_index[word] if (word != None) and (not disableDurations) and (not useSingleValueForDuration) else None for word in durations]
        eventAttributeClusters = [word_to_index[word] if (word != None) and (not disableEventAttributes) else None for word in eventAttributeClusters]
        return (labels, durations, eventAttributeClusters, durationValues)
#            self.activitiesForPrediction[key] = (labels, durations, eventAttributeClusters)
#        return self.activitiesForPrediction[key]

    def getCaseAttributeWordIndexes(self, word_to_index):
        if self.indexedCaseAttributeWords == None:
            self.indexedCaseAttributeWords = [(word_to_index[val] if val in word_to_index else None) for val in self.filteredCaseAttributes]
        return self.indexedCaseAttributeWords

    def getEventAttributeWordIndexes(self, word_to_index):
        if self.indexedEventAttributeWords == None:
            self.indexedEventAttributeWords = [[(word_to_index[val] if val in word_to_index else None) for val in fea] for fea in self.filteredEventAttributes]
        return self.indexedEventAttributeWords

def print_trace(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:-1]]
    writeLog(" ".join(sentence_str))
    sys.stdout.flush()

def generate_trace(model, index_to_word, word_to_index, min_length=5):
    # We start the sentence with the start token
    new_sentence = []
    # Repeat until we get an end token
    selIndex = word_to_index[IN_SELECTION_TOKEN]
    notSelIndex = word_to_index[NOT_IN_SELECTION_TOKEN]

    while not ((len(new_sentence) > 0) and ((new_sentence[-1] == selIndex) or (new_sentence[-1] == notSelIndex))):
        next_word_probs = model.predict(new_sentence)[-1]
        samples = np.random.multinomial(1, next_word_probs)
        sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
        # Seomtimes we get stuck if the sentence becomes too long, e.g. "........" :(
        # And: We don't want sentences with UNKNOWN_TOKEN's
        if len(new_sentence) > 100 or sampled_word == word_to_index[UNKNOWN_TOKEN]:
            return None
    if len(new_sentence) < min_length:
        return None
    return new_sentence

def generate_traces(model, n, index_to_word, word_to_index):
    for i in range(n):
        sent = None
        while not sent:
            sent = generate_trace(model, index_to_word, word_to_index)
        print_trace(sent, index_to_word)

def predict_outcome(model, test, word_to_index):
    nextPrediction = model.predict(test)[-1]
    selIndex = word_to_index[IN_SELECTION_TOKEN]
    notSelIndex = word_to_index[NOT_IN_SELECTION_TOKEN]
    selProb = nextPrediction[selIndex]
    notSelProb = nextPrediction[notSelIndex]
    return selProb >= notSelProb

def get_filename(figure_type, name, file_type, output_path = None):
    dtstr = datetime.now().replace(microsecond=0).isoformat().replace("-", "").replace(":", "")
    return (output_path if output_path != None else _output_path)  + figure_type + "-" + name + "-" + dtstr + "." + file_type

_output_path = ""
_input_files_path = ""
_log_filename = ""
_results_filename = ""
_log_to_file_only = False

def getOutputPath():
    return _output_path

def getInputPath():
    return _input_files_path

def configure(input_files_path, output_path, log_to_file_only):
    global _output_path
    global _input_files_path
    global _log_filename
    global _results_filename
    global _log_to_file_only
    _output_path = output_path
    _input_files_path = input_files_path
    _log_filename = get_filename("log", "", "txt")
    _results_filename = get_filename("results", "", "csv")
    _log_to_file_only = log_to_file_only
    with open(_results_filename, "w", newline="", encoding='utf-8-sig') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["Time", "Status", "Name", "TestName", "Dataset", "CVRunId", "TrainDatasetSize", "TestDatasetSize", "DatasetSize", "Algorithm", "NumLayers", "HiddenDimSize", "Optimizer", "LearningRate", "SeqLength", "InputVectorSize", "BatchSize", "GradClipping", "ItemsBetween", "BestModelIteration", "TestIteration", "Iteration", "Epoch", "TrainDataInitTimeUsed", "TrainLayerInitTimeUsed", "CumulExactTrainTimeUsed", "TimeUsed", "CumulTimeUsed", "TimeUsedForTest", "CumulTimeUsedForTest", "SR_Train", "SR_Test", "SR_Test75p", "SR_Test50p", "SR_Test25p", "AvgCost", "AUC", "TP", "TN", "FP", "FN", "AllConfusions", "PredictOnlyOutcome", "FinalTraceOnly", "TraceLengthMod", "FixedLength", "MaxNumActivities", "TruncateUnknowns", "ActivityLabels", "Durations", "EventAttributes", "CaseAttributes", "RawEventAttributes", "RawCaseAttributes", "PredictNextActivity", "SingleEventClustering", "DurationSplitMethod", "CaseClusteringMethod", "EventClusteringMethod", "CaseClusteringIncludeActivityOccurrences", "CaseClusteringIncludeCaseAttributes", "IncludeActivityOccurrencesAsRawCaseAttributes", "UseSingleValueForDuration", "MaxNumCaseClusters", "MaxNumEventClusters", "MinimumUsageForCaseAttributes", "MinimumUsageForEventAttributes"])

def getInputDatasetFilename(dataset_name):
    return _input_files_path + dataset_name + ".json"

def writeLog(message):
    global _log_to_file_only
    message = datetime.now().replace(microsecond=0).isoformat() + " \t" + message
    if (not _log_to_file_only):
        print(message)
    with open(_log_filename, "a", encoding='utf-8-sig') as logfile:
        logfile.write(message + "\n")

def writeResultRow(cells):
    with open(_results_filename, "a", newline="", encoding='utf-8-sig') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(cells)

def writeTestResultRow(cells):
    writeResultRow(cells)

def encodeCaseAttributeValueForVocabulary(val, attributeId):
    return "%sC%d_%s" % (ATTRIBUTE_COLUMN_PREFIX, attributeId, val)
def encodeEventAttributeValueForVocabulary(val, attributeId):
    return "%sE%d_%s" % (ATTRIBUTE_COLUMN_PREFIX, attributeId, val)
