#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 11:59:43 2017

Test framework sources used to perform the tests required by paper: 
"hinkka"
by Markku Hinkka, Teemu Lehto and Keijo Heljanko
"""
# pip install pyclustering
# conda install -c conda-forge matplotlib
# conda install pillow

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
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import metrics
from time import time
from pathlib import Path
import pandas as pd
import nltk
import itertools
import json
from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES;
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer;
from pyclustering.cluster.xmeans import xmeans, splitting_type;

#from pyxmeans import _minibatch
#from pyxmeans.mini_batch import MiniBatch
#from pyxmeans.xmeans import XMeans

from my_utils import encodeCaseAttributeValueForVocabulary, encodeEventAttributeValueForVocabulary, TraceData, writeLog, writeResultRow, get_filename, getOutputPath, ATTRIBUTE_COLUMN_PREFIX, OUTCOME_SELECTION_TOKEN_PREFIX, DURATION_TOKEN_PREFIX, EVENT_ATTRIBUTE_TOKEN_PREFIX, WORD_PART_SEPARATOR, CASE_ATTRIBUTE_TOKEN_PREFIX, UNKNOWN_TOKEN, OTHER_TOKEN

pd.options.mode.chained_assignment = None  # default='warn'

def train_hashvalue(df, parameters):
    hashes = {}
    hashId = 0
    nextHashId = 0
    labels = []
    for row in df:
        hashValue = hash(tuple(row))
        if (hashValue in hashes):
            hashId = hashes[hashValue]
        else:
            nextHashId += 1
            hashId = hashes[hashValue] = nextHashId
        labels.append(hashId)
    writeLog("Hashvalue clustering resulted into %d unique hash values for %d rows." % (len(hashes), len(labels)))
    return hashes, labels, [i for i in range(nextHashId)]

def predict_hashvalue(df, model):
    labels = []
    for row in df:
        hashId = hash(tuple(row))
        labels.append(model[hashId] if (hashId in model) else None)
    return labels


def train_kmeans(df, parameters):
    num_clusters = parameters["num_clusters"]
    return do_train_kmeans(df, num_clusters)

def do_train_kmeans(df, num_clusters, centers = None):
    if (df.shape[1] == 0) or (num_clusters < 2):
        writeLog("No columns in the table to be clustered. Returning constant labels.")
        model = None
        labels = len(df) * [0]
        return model, labels, [0]

    if centers == None:
        model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1,
                                init_size=1000, batch_size=1000, verbose=False)
    else:
        model = KMeans(n_clusters=num_clusters, init=np.asarray(centers), n_init=1, max_iter=1)
    x = model.fit(df)
    writeLog("K-means model created for %d clusters." % (model.n_clusters))
    return model, x.labels_, [i for i in range(model.n_clusters)]

def predict_kmeans(df, model):
    if model == None:
        return len(df) * [0]
    return model.predict(df)

def train_xmeans(df, parameters):
    # create object of X-Means algorithm that uses CCORE for processing
    # initial centers - optional parameter, if it is None, then random centers will be used by the algorithm.
    # let's avoid random initial centers and initialize them using K-Means++ method:
    max_num_clusters = parameters["max_num_clusters"]
    num_clusters = parameters["num_clusters"]
    initial_centers = kmeans_plusplus_initializer(df, min(df.shape[0], num_clusters)).initialize()
    print("CLUSTERING DF: ", df)
    print("CLUSTERING DF shape: ", df.shape)
    # NaNs on the numpy matrix give segmentation fault from the pyclustering xmeans
    df = np.nan_to_num(df)
    xmeans_instance = xmeans(df, initial_centers, ccore=True, kmax=num_clusters)

    # run cluster analysis
    xmeans_instance.process()

    # obtain results of clustering
    clusters = xmeans_instance.get_clusters()
    writeLog("X-means clustered using %d clusters (init: %d, max: %d). Using that as the desired number of clusters for k-means." % (len(clusters), num_clusters, max_num_clusters))
    return do_train_kmeans(df, len(clusters), xmeans_instance.get_centers())

#    result = [0 for x in range(len(df))]
#    for clusterId, rows in enumerate(clusters):
#        for rowId in rows:
#            result[rowId] = clusterId
#    return xmeans_instance, result


#    num_clusters = parameters["num_clusters"]
    # create instance of Elbow method using K value from 1 to 10.
#    kmin, kmax = 1, 20
#    elbow_instance = elbow(df, kmin, kmax)
    # process input data and obtain results of analysis
#    elbow_instance.process()
#    num_clusters = elbow_instance.get_amount()   # most probable amount of clusters
    # https://datascience.stackexchange.com/questions/34187/kmeans-using-silhouette-score

def predict_xmeans(df, model):
    if model == None:
        return len(df) * [0]
    df = np.nan_to_num(df)
    return model.predict(df)

def train_skmeans(df, parameters):
#    num_clusters = parameters["num_clusters"]
    # create instance of Elbow method using K value from 1 to 10.
#    kmin, kmax = 1, 20
#    elbow_instance = elbow(df, kmin, kmax)
    # process input data and obtain results of analysis
#    elbow_instance.process()
#    num_clusters = elbow_instance.get_amount()   # most probable amount of clusters
    # https://datascience.stackexchange.com/questions/34187/kmeans-using-silhouette-score

    max_num_clusters = parameters["max_num_clusters"]
    Ks = range(2, min(max_num_clusters, len(df)) + 1)
    kms = [MiniBatchKMeans(n_clusters=i, init='k-means++', n_init=1,
                            init_size=1000, batch_size=1000, verbose=False) for i in Ks]
    writeLog("Performing K-means for cluster sizes 2 - %d" % (min(max_num_clusters, len(df))))
    sil_coeff = []
    all_labels = []
    distance_matrix = None
    max_num_samples_training_cluster = parameters["max_num_samples_training_cluster"]
    if len(df) > max_num_samples_training_cluster:
        writeLog("The number of samples to be clustered (%d) exceeds the configured maximum of %d. Taking random sample of the configured maximum size." % (len(df), max_num_samples_training_cluster))
        traindf = df[np.random.choice(df.shape[0], max_num_samples_training_cluster, replace=False), :]
    else:
        traindf = df
    for i, km in enumerate(kms):
        x = km.fit(traindf)
        if (i == 0):
            distance_matrix = pairwise_distances(traindf, metric="euclidean")
        score = 0.0
        try:
            score = silhouette_score(distance_matrix, x.labels_, metric='precomputed')
            writeLog("sihouette_score for cluster size %d = %f" % (km.n_clusters, score))
        except:
            writeLog("Unable to calculate sihouette_score for cluster size %d. Using %f." % (km.n_clusters, score))
        if len(traindf) < len(df):
            labels = km.predict(df)
        else:
            labels = x.labels_
        sil_coeff.append(score)
        all_labels.append(labels)
        if score >= 1.0:
            writeLog("Maximum silhouette score reached. No need to consider any more clusters.")
            break
    max_index = np.asarray(sil_coeff).argmax(axis=0)
    model = kms[max_index]
    labels = all_labels[max_index]
    writeLog("Optimum number of clusters: " + str(model.n_clusters))
    return model, labels, [i for i in range(model.n_clusters)]

    # create object of X-Means algorithm that uses CCORE for processing
    # initial centers - optional parameter, if it is None, then random centers will be used by the algorithm.
    # let's avoid random initial centers and initialize them using K-Means++ method:
#    initial_centers = kmeans_plusplus_initializer(df, num_clusters).initialize();
#    xmeans_instance = xmeans(df, initial_centers, ccore=True);

    # run cluster analysis
#    xmeans_instance.process();

    # obtain results of clustering
#    clusters = xmeans_instance.get_clusters();
#    result = [0 for x in range(len(df))]
#    for clusterId, rows in enumerate(clusters):
#        for rowId in rows:
#            result[rowId] = clusterId
#    return xmeans_instance, result

def predict_skmeans(df, model):
    if model == None:
        return len(df) * [0]
    return model.predict(df)

algorithms = {
    "hashvalue": { 
        "train": train_hashvalue,
        "predict": predict_hashvalue
    },
    "kmeans": { 
        "train": train_kmeans,
        "predict": predict_kmeans
    },
    "s+kmeans": { 
        "train": train_skmeans,
        "predict": predict_skmeans
    },
    "xmeans": { 
        "train": train_xmeans,
        "predict": predict_xmeans
    }
}

class Clustering:
    def __init__(self, algorithm = None, globalParameters = None, parameters = None, copyFrom = None):
        if copyFrom != None:
            self.algorithm = copyFrom.algorithm
            self.parameters = dict(copyFrom.parameters)
        else:
            self.algorithm = algorithm
            if (globalParameters != None):
                self.parameters = dict(globalParameters)
                self.parameters.update(parameters)
            else:
                self.parameters = parameters
        writeLog("Creating new clustering object for algorithm: " + self.algorithm)
        self.model = None
        self.vectorizer = None
        self.known_values = None
        self.labels = []

    def getCaseFeatureGroupsToInclude(self):
        ica_clustering = (not self.parameters["disable_case_attributes"]) and (self.parameters["case_clustering_include_case_attributes"])
        iao_clustering = (not self.parameters["disable_case_attributes"]) and (self.parameters["case_clustering_include_activity_occurrences"])
        ica_filtering = ica_clustering or (not self.parameters["disable_raw_case_attributes"])
        iao_filtering = iao_clustering or ((not self.parameters["disable_raw_case_attributes"]) and (self.parameters["include_activity_occurrences_as_raw_case_attributes"]))
        return ica_clustering, iao_clustering, ica_filtering, iao_filtering

    def trainForCaseClustering(self, eventlog, cases):
        if self.parameters["disable_case_attributes"] and self.parameters["disable_raw_case_attributes"]:
            writeLog("Case clustering not needed. Skipping it.")
            for t in cases:
                t["_cluster"] = 0
            return
        writeLog("Clustering %d cases" % (len(cases)))
#        num_clusters = self.parameters["num_clusters"]
#        if (num_clusters <= 1):
#            for t in cases:
#                t["_cluster"] = 0
#            return

        t0 = time()

        data = []
        cols = []
        ica_clustering, iao_clustering, ica_filtering, iao_filtering = self.getCaseFeatureGroupsToInclude()
        ica_cols = []
        iao_cols = []
        if ica_filtering:
            data += [c["a"] + c["occ"] for c in cases] if iao_filtering else [c["a"] for c in cases]
            ica_cols = ["A_" + a for a in eventlog.data["attributes"]["case"]]
            cols += ica_cols
        if iao_filtering:
            if not ica_filtering:
                data += [c["occ"] for c in cases]
            iao_cols = ["O_" + a["name"] for a in eventlog.data["activities"]]
            cols += iao_cols
        df = pd.DataFrame(data, columns=cols)
        self.known_values = self.filterUnusualValues(df, self.parameters)
        if (ica_filtering and (not ica_clustering)):
            df = df.drop(ica_cols, axis = 1)
        if (iao_filtering and (not iao_clustering)):
            df = df.drop(iao_cols, axis = 1)

        if ("Cost" in df.columns):
            df = df.drop(["Cost"], axis = 1)
        if ("_cluster" in df.columns):
            df = df.drop(["_cluster"], axis = 1)

        if not self.parameters["disable_case_attributes"]:
            self.model, self.vectorizer, labels = self.train(df, self.parameters)
            for i, d in enumerate(labels):
                cases[i]["_cluster"] = d
            writeLog("Case clustering done in %0.3fs" % (time() - t0))
        else:
            self.model = None
            self.vectorizer = None
            writeLog("Case data filtering done in %0.3fs" % (time() - t0))


    def trainForEventClustering(self, eventlog, cases):
        if self.parameters["disable_event_attributes"] and self.parameters["disable_raw_event_attributes"]:
            writeLog("Event clustering not needed. Skipping it.")
            for c in cases:
                for e in c["t"]:
                    e.append(0)
            return
        writeLog("Clustering events in %d cases" % (len(cases)))
#        num_clusters = self.parameters["num_clusters"]
#        if (num_clusters <= 1):
#            for c in cases:
#                for e in c["t"]:
#                    e.append(0)
#            return

        t0 = time()

        if (self.parameters["use_single_event_clustering"]):
            events = []
            for c in cases:
                for e in c["t"]:
                    events.append(["" if i == None else i for i in e[2:]])
            df = pd.DataFrame(events, columns=eventlog.data["attributes"]["event"])
            known_values = self.filterUnusualValues(df, self.parameters)
            if not self.parameters["disable_event_attributes"]:
                model, vectorizer, labels = self.train(df, self.parameters)
                i = 0
                for c in cases:
                    for e in c["t"]:
                        e.append(labels[i])
                        i += 1
                self.vectorizer = { "primary": vectorizer }
                self.model = { "primary": model }
            else:
                model = None
                vectorizer = None
            self.known_values = { "primary": known_values }
        else:
            self.model = {}
            self.vectorizer = {}
            self.known_values = {}

            eventAttributes = eventlog.data["attributes"]["event"]

            activities = eventlog.getActivityOccurrences(cases)
            for activityId, activity in activities.items():
                t0 = time()
                writeLog("Clustering %d events for activity: %s (id: %s)" % (len(activity["occ"]), activity["name"], activityId))

                events = [None] * len(activity["occ"])
                maxLen = len(eventAttributes) + 2
                for i, e in enumerate(activity["occ"]):
                    events[i] = e[2:maxLen]

                if (len(events) < 1):
                    i = 0
                    for e in activity["occ"]:
                        e.append(0)
                        i += 1
                    continue
            
                df = pd.DataFrame(events, columns=eventlog.data["attributes"]["event"])
                self.known_values[activityId] = self.filterUnusualValues(df, self.parameters)

                if not self.parameters["disable_event_attributes"]:
                    self.model[activityId], self.vectorizer[activityId], labels = self.train(df, self.parameters)
                    i = 0
                    if not self.parameters["disable_event_attributes"]:
                        for e in activity["occ"]:
                            e.append(labels[i])
                            i += 1
                else:
                    self.model[activityId] = None
                    self.vectorizer[activityId] = None

        writeLog("Event clustering done in %0.3fs" % (time() - t0))

    def filterUnusualValues(self, df, parameters):
        writeLog("Number of colums to filter unusual values from %d" % (len(df.columns)))

        t0 = time()

        threshold = parameters["ignore_values_threshold"] * len(df)
        known_values = {}
        for col in df.columns:
            writeLog("Replacing unusual values in column '%s' with minimum usage of %d rows." % (col, threshold))
            vc = df[col].value_counts()
            toRemove = vc[vc <= threshold].index
            toKeep = vc[vc > threshold].index
            known_values[col] = toKeep
            writeLog("Remaining known values: %s (removed %d values out of %d values)" % (str([i for i in toKeep]), len(toRemove), len(toKeep)))
            if len(toRemove) > 0:
                df[col].replace(toRemove, OTHER_TOKEN, inplace=True)
        writeLog("Unusual value filtering done in %f s" % (time() - t0))
        return known_values

    def train(self, df, parameters):
        writeLog("Number of colums to cluster %d" % (len(df.columns)))

        t0 = time()

        vectorizer = DictVectorizer(sparse = False, dtype=np.float32)
        writeLog("Vectorizing data frame of shape: %s" % (str(df.shape)))
        X = vectorizer.fit_transform(df.to_dict(orient = 'records'))

        writeLog("Data vectorization done in %fs" % (time() - t0))
        writeLog("n_samples: %d, n_features: %d" % X.shape)

        t0 = time()
        alg = algorithms[self.algorithm]

        # #############################################################################
        # Do the actual clustering
        if df.shape[0] < 2:
            writeLog("One row or less to cluster. Returning constant labels.")
            model = None
            labels = len(df) * [0]
            allLabels = [0]
        elif df.shape[1] == 0:
            writeLog("No columns in the table to be clustered. Returning constant labels.")
            model = None
            labels = len(df) * [0]
            allLabels = [0]
        else:
            model, labels, allLabels = alg["train"](X, parameters)

        if (len(allLabels) > len(self.labels)):
            self.labels = allLabels

        writeLog("Clustering using %s done in %fs" % (self.algorithm, time() - t0))
        return model, vectorizer, labels

    def getCaseClusteringDataFrame(self, eventlog, cases):
        cols = []
        ica = self.parameters["case_clustering_include_case_attributes"]
        iao = self.parameters["case_clustering_include_activity_occurrences"]
        if ica:
            cols += eventlog.data["attributes"]["case"]
        if iao:
            cols += [a["name"] for a in eventlog.data["activities"]]
        rows = [([] + (c["occ"] if iao else []) + (["" if i == None else i for i in c["a"]] if ica else [])) for c in cases]
        return pd.DataFrame(rows, columns=cols)

    def clusterCases(self, eventlog, testData):
        if self.model == None:
            for td in testData:
                td["_cluster"] = 0
            return

        df = self.getCaseClusteringDataFrame(eventlog, testData)

        labels = self.predict(df, self.model, self.vectorizer, self.known_values)

        for i, d in enumerate(labels):
            testData[i]["_cluster"] = d

    def clusterEvents(self, eventlog, testData):
        if self.model == None:
            for c in testData:
                for e in c["t"]:
                    e.append(0)
            return

        if (self.parameters["use_single_event_clustering"]):
            self.clusterTestDataUsingSingleClustering(eventlog, testData)
        else:
            self.clusterTestDataUsingMultipleClusterings(eventlog, testData)

    def clusterTestDataUsingSingleClustering(self, eventlog, testData):
        eventAttributes = eventlog.data["attributes"]["event"]

        events = []
        for c in testData:
            for e in c["t"]:
#                if (len(e) > len(eventAttributes) + 2):
#                    del(e[len(e) - 1])
                events.append(["" if i == None else i for i in e[2:]])
        df = pd.DataFrame(events, columns=eventAttributes)
        labels = self.predict(df, self.model["primary"], self.vectorizer["primary"], self.known_values["primary"])

        i = 0
        for c in testData:
            for e in c["t"]:
                e.append(labels[i])
                i += 1

    def clusterTestDataUsingMultipleClusterings(self, eventlog, testData):
        num_clusters = self.parameters["num_clusters"]
        eventAttributes = eventlog.data["attributes"]["event"]

        activities = eventlog.getActivityOccurrences(testData)
        for activityId, activity in activities.items():
            numEvents = len(activity["occ"])
            writeLog("Clustering %d test events for activity: %s (id: %s)" % (numEvents, activity["name"], activityId));
            model = self.model[activityId] if activityId in self.model else None

            if ((numEvents < 2) or (model == None)):
                i = 0
                for e in activity["occ"]:
                    e.append(0)
                    i += 1
                continue

            events = [None] * numEvents
            maxLen = len(eventAttributes) + 2
            for i, e in enumerate(activity["occ"]):
                events[i] = e[2:maxLen]

            df = pd.DataFrame(events, columns=eventlog.data["attributes"]["event"])
            labels = self.predict(df, model, self.vectorizer[activityId], self.known_values[activityId])

            i = 0
            for e in activity["occ"]:
                e.append(labels[i])
                i += 1

    def predict(self, df, model, vectorizer, known_values):
        threshold = self.parameters["ignore_values_threshold"] * len(df)
        if threshold > 0:
            for col in df.columns:
                writeLog("Replacing unusual values in column %s." % (col))
                if col in known_values:
                    isin = df[col].isin(known_values[col])
                    df[col].loc[-isin] = OTHER_TOKEN

        writeLog("Vectorizing data frame of shape: %s" % (str(df.shape)))
        XX = vectorizer.transform(df.to_dict(orient = 'records'))
        alg = algorithms[self.algorithm]
        return alg["predict"](XX, model)

    def filterEventAttributes(self, eventlog, trace):
        result = []
        eventAttributes = eventlog.data["attributes"]["event"]
        if (self.parameters["use_single_event_clustering"]):
            for eventId, e in enumerate(trace.eventAttributes):
                r = []
                kv = self.known_values["primary"]
                for attributeId, val in enumerate(e[0:len(eventAttributes)]):
                    attributeName = eventAttributes[attributeId]
                    valueIndex = kv[attributeName]
                    v = val if (val in valueIndex) else OTHER_TOKEN
                    r.append(encodeEventAttributeValueForVocabulary(v, attributeId))
                result.append(r)
        else:
            for eventId, e in enumerate(trace.eventAttributes):
                r = []
                activity = eventlog.activitiesByLabel[trace.activityLabels[eventId]]
                activityId = activity["id"]
                if activityId in self.known_values:
                    kv = self.known_values[activityId]
                    for attributeId, val in enumerate(e[0:len(eventAttributes)]):
                        attributeName = eventAttributes[attributeId]
                        valueIndex = kv[attributeName]
                        v = val if (val in valueIndex) else OTHER_TOKEN
                        r.append(encodeEventAttributeValueForVocabulary(v, attributeId))
                else:
                    for attributeId, val in enumerate(e):
                        r.append(encodeEventAttributeValueForVocabulary(OTHER_TOKEN, attributeId))
                result.append(r)
        trace.filteredEventAttributes = result

    def filterCaseAttributes(self, eventlog, trace):
        result = []
        caseAttributes = eventlog.data["attributes"]["case"]
        activities = eventlog.data["activities"]
        ica_clustering, iao_clustering, ica_filtering, iao_filtering = self.getCaseFeatureGroupsToInclude()
        lastAttributeId = 0
        if ica_filtering:
            lastAttributeId = len(caseAttributes)
        kv = self.known_values
        for attributeId, val in enumerate(trace.caseAttributes):
            if (attributeId < lastAttributeId):
                attributeName = "A_" + caseAttributes[attributeId]
                valueIndex = kv[attributeName]
            else:
                attributeName = "O_" + activities[attributeId - lastAttributeId]["name"]
                valueIndex = kv[attributeName]
            v = val if (val in valueIndex) else OTHER_TOKEN
            result.append(encodeCaseAttributeValueForVocabulary(v, attributeId))
        trace.filteredCaseAttributes = result

    def getSetOfKnownCaseAttributeValuesForVocabulary(self, eventlog, traces):
        result = []
        if self.parameters["disable_raw_case_attributes"]:
            return result
        caseAttributes = eventlog.data["attributes"]["case"]
        kv = self.known_values
        for attributeId, attributeName in enumerate(caseAttributes):
            r = set([encodeCaseAttributeValueForVocabulary(OTHER_TOKEN, attributeId)])
            attributeKey = "A_" + attributeName
            vals = kv[attributeKey]
            r.update([encodeCaseAttributeValueForVocabulary(v, attributeId) for v in vals])
            result.append(r)

        if not self.parameters["include_activity_occurrences_as_raw_case_attributes"]:
            return [item for sublist in result for item in sublist]

        activities = eventlog.data["activities"]
        for activityId, activity in enumerate(activities):
            attributeId = activityId + len(caseAttributes)
#            r = set([encodeCaseAttributeValueForVocabulary(OTHER_TOKEN, attributeId)])
            r = set()
            attributeKey = "O_" + activity["name"]
            vals = kv[attributeKey]
            r.update([encodeCaseAttributeValueForVocabulary(v, attributeId) for v in vals])
            result.append(r)
        return [item for sublist in result for item in sublist]
        
    def getSetOfKnownEventAttributeValuesForVocabulary(self, eventlog, traces):
        result = []
        if self.parameters["disable_raw_event_attributes"]:
            return result
        result = []
        eventAttributes = eventlog.data["attributes"]["event"]
        if (self.parameters["use_single_event_clustering"]):
            kv = self.known_values["primary"]
            for attributeId, e in enumerate(eventAttributes):
#                r = set([encodeEventAttributeValueForVocabulary(OTHER_TOKEN, attributeId)])
                attributeName = eventAttributes[attributeId]
                r = set()
                vals = kv[attributeName]
                r.update([encodeEventAttributeValueForVocabulary(v, attributeId) for v in vals])
                result.append(r)
        else:
            for attributeId, e in enumerate(eventAttributes):
#                r = set([encodeEventAttributeValueForVocabulary(OTHER_TOKEN, attributeId)])
                attributeName = eventAttributes[attributeId]
                r = set()
                for activity in eventlog.data["activities"]:
                    activityId = activity["id"]
                    if activityId in self.known_values:
                        kv = self.known_values[activityId]
                        vals = kv[attributeName]
                    else:
                        vals = []
                    r.update([encodeEventAttributeValueForVocabulary(v, attributeId) for v in vals])
                result.append(r)
        return [item for sublist in result for item in sublist]

    def addUndefinedParameters(self, parameters):
        for key, value in parameters.items():
            if not key in self.parameters:
                self.parameters[key] = value

    def getClusterLabels(self):
        return self.labels

    def save(self):
        saved = {
            "algorithm": self.algorithm,
            "model": self.model,
            "vectorizer": self.vectorizer,
            "parameters": self.parameters,
            "known_values": self.known_values,
            "labels": self.labels
        }
        return saved

    def load(self, saved):
        self.algorithm = saved["algorithm"]
        self.model = saved["model"]
        self.vectorizer = saved["vectorizer"]
        self.parameters = saved["parameters"]
        self.known_values = saved["known_values"]
        self.labels = saved["labels"]
