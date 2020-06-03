#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 11:59:43 2017

Test framework sources used to perform the tests required by paper: 
"hinkka"
by Markku Hinkka, Teemu Lehto and Keijo Heljanko
"""
import sys
import os
import numpy as np
import json
import logging
import traceback
from time import time, sleep
from optparse import OptionParser
from pathlib import Path
import collections
import datetime
import copy
from multiprocessing import Process

from model import Model
from modelcluster import ModelCluster
from eventlog import EventLog
from cluster import Clustering
from my_utils import writeLog, generate_traces, configure, TraceData, get_filename, getInputDatasetFilename, \
    getOutputPath, getInputPath
import theano

outputDirectory = "output"
# outputDirectory = "C:/Users/marhink/Dropbox/Aalto/testing/testruns/"
# outputDirectory = "d:\\dev\\aalto\\testing\\testruns\\"
inputFilesDirectory = "testdata/"
modelOutputDirectory = inputFilesDirectory + "models"

# http://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
# op.add_option("--built-in-test",
#              action="store_false", dest="built_in_test", default=True,
#              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")
op.add_option("--log_to_file_only",
              action="store_true", dest="log_to_file_only", default=False,
              help="Output log only into a logfile.")
op.add_option("--input_data_from_standard_input2",
              action="store_true", dest="input_data_from_standard_input2", default=False,
              help="Read input data from standard input.")
op.add_option("--log_to_file_only2",
              action="store_true", dest="log_to_file_only2", default=False,
              help="Output log only into a logfile.")
op.add_option("--input_data_from_standard_input",
              action="store_true", dest="input_data_from_standard_input", default=False,
              help="Read input data from standard input.")
op.add_option("--configuration_from_standard_input",
              action="store_true", dest="configuration_from_standard_input", default=False,
              help="Read configuration from standard input.")
op.add_option("-m", "--model", dest="model_filename",
              help="File containing trained model", metavar="FILE")
op.add_option("-s", "--skip", dest="skip_tests", type="int", default=0,
              help="Number of tests to skip from the beginning")
op.add_option("--m2", dest="model_filename2",
              help="File containing trained model", metavar="FILE")
op.add_option("-i", "--input", dest="input_filename",
              help="File containing trace data of traces to use for training", metavar="FILE")
op.add_option("-t", "--test", dest="test_filename",
              help="File containing trace data of traces to use for testing", metavar="FILE")
op.add_option("--i2", dest="input_filename2",
              help="File containing trace data of traces to use for training", metavar="FILE")
op.add_option("-c", "--configuration", dest="configuration_filename",
              help="File containing configuration for the tests", metavar="FILE")
op.add_option("-o", "--output", dest="output_filename",
              help="Name of the file to generate", metavar="FILE")
op.add_option("--predict_next_activity",
              action="store_true", dest="predict_next_activity", default=False,
              help="Predict the next activity.")
op.add_option("--disable_durations",
              action="store_true", dest="disable_durations", default=False,
              help="Do not use duration information in training and prediction.")
op.add_option("--disable_event_attributes",
              action="store_true", dest="disable_event_attributes", default=False,
              help="Do not use event attributes in training and prediction.")
op.add_option("--disable_case_attributes",
              action="store_true", dest="disable_case_attributes", default=False,
              help="Do not use case attributes in training and prediction.")
op.add_option("--disable_activity_labels",
              action="store_true", dest="disable_activity_labels", default=False,
              help="Do not use activity labels in training and prediction.")
op.add_option("-w", "--wait-for-config", dest="test_config_filename",
              help="File to use to bring new test configurations", metavar="FILE")


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)

print("OPTIONS: ", opts)
print("ARGS: ", args)

if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

random_seed = 123

default_parameters = {
    "algorithm": "gru",
    "case_name": "test",
    "dataset_name": None,
    "test_name": "",
    "num_layers": 1,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "num_callbacks": 100,
    "batch_size": 256,
    "hidden_dim_size": 256,
    # "batch_size": 16,
    # "hidden_dim_size": 8,
    "num_iterations_between_reports": 1000,
    "grad_clipping": 100,
    "predict_only_outcome": False,
    "final_trace_only": True,
    "trace_length_modifier": 1.0,
    "truncate_unknowns": False,
    "max_num_words": None,
    "num_epochs_per_iteration": 1.0,
    "num_models": 1,
    "file_handle": "test",
    "use_single_event_clustering": False,
    "num_case_clusters": 5,
    "num_event_clusters": 5,
    "case_clustering_method": "xmeans",
    "case_clustering_include_activity_occurrences": False,
    "case_clustering_include_case_attributes": True,
    "event_clustering_method": "xmeans",
    "ignore_values_threshold_for_case_attributes": 0.1,
    "ignore_values_threshold_for_event_attributes": 0.1,
    "duration_split_method": "5-buckets",
    "predict_next_activity": opts.predict_next_activity,
    "disable_durations": opts.disable_durations,
    "disable_event_attributes": opts.disable_event_attributes,
    "disable_case_attributes": opts.disable_case_attributes,
    "model_filename": opts.model_filename,
    "input_filename": opts.input_filename,
    "test_filename": opts.test_filename,
    "output_filename": opts.output_filename,
    "test_config_filename": opts.test_config_filename,
    "disable_raw_event_attributes": False,
    "disable_raw_case_attributes": False,
    "include_activity_occurrences_as_raw_case_attributes": False,
    "dataset_name": "",
    "write_input_to_file": False,
    "predict_only": False,
    "max_num_samples_training_cluster": 10000000,
    "max_num_traces_to_test": 100000000,
    "max_num_traces_in_testing": 100000000,
    "use_single_value_for_duration": False,
    "max_num_case_clusters": 20,
    "max_num_event_clusters": 20,
    "pause_filename": "pause.txt",
    "input_directory": inputFilesDirectory,
    "output_directory": outputDirectory,
    "model_output_directory": modelOutputDirectory,
    "spawn_worker_processes": False,
    "max_num_cases_in_training": None,
    "max_num_traces_in_training": None,
    "max_num_traces_in_training_test": None,
    "test_data_percentage": 0.75,
    "split_traces_to_prefixes": False,
    "min_splitted_trace_prefix_length": 1, # Use prefixes of length 1 and onwards
    "max_trace_length": 10000,
    "cross-validation-splits": None,
    "create-unknown-tokens": False
}

configuration = {
    "for_each": [
        {
            "input_filename": "BPIC13_incidents-ne-full",
            "dataset_name": "bpic13"
        }
    ],
    "runs": [
        {
            "case_name": "both-event-attributes",
            "disable_raw_event_attributes": False,
            "disable_event_attributes": False
        }
    ],
}

configurationPath = None


def run(parameters):
    rng = np.random.RandomState(random_seed)

    writeLog("Running test using parameters: " + json.dumps(parameters))

    inputJson = None
    if (opts.input_data_from_standard_input):
        writeLog("Reading from standard input")
        inputJson = sys.stdin.readline()
        writeLog("Standard input reading finished")
        if (parameters["write_input_to_file"]):
            filename = get_filename("testdata_", "%s_%s_%s" % (parameters["file_handle"], "", ""), "json")
            with open(filename, "w") as f:
                f.write(inputJson)

    if (parameters["model_filename"] != None):
        m = ModelCluster(rng)
        print("Parameters model filename: ", parameters["model_filename"])
        m.load(parameters["model_filename"], parameters)
        inputFilename = None if parameters["test_filename"] == None else parameters["test_filename"]
        if (inputFilename != None):
            writeLog("Reading test data from file: " + inputFilename)
        el = EventLog(parameters, rng, inputFilename, modelCluster=m, inputJson=inputJson)
        jsonResult = "{}"
        if (len(el.testData) > 0):
            writeLog("Test set contains %d cases." % (len(el.testData)))
            result = m.test(el)
            jsonResult = json.dumps(result)
            filename = get_filename("predict_result",
                                    "%s_%s_%s" % (parameters["file_handle"], m.case_name, m.eventlog.filename), "json")
            with open(filename, "w") as f:
                f.write(jsonResult)
            writeLog("Generated results saved into file: %s" % filename)
        else:
            writeLog("Test set is empty. No results created.")
        print(jsonResult)
    elif ((parameters["input_filename"] != None) or (inputJson != None)):
        if parameters["cross-validation-splits"] != None:
            EventLog.performCrossValidatedTests(parameters, inputJson, rng)
            return
        e = EventLog(parameters, rng, parameters["input_filename"], parameters["test_data_percentage"],
                     inputJson=inputJson)
        m = ModelCluster(rng)
        m.initialize(
            parameters=parameters,
            case_clustering=Clustering(parameters["case_clustering_method"], parameters, {
                "num_clusters": parameters["num_case_clusters"],
                "max_num_clusters": parameters["max_num_case_clusters"],
                "ignore_values_threshold": parameters["ignore_values_threshold_for_case_attributes"]
            }),
            event_clustering=Clustering(parameters["event_clustering_method"], parameters, {
                "num_clusters": parameters["num_event_clusters"],
                "max_num_clusters": parameters["max_num_event_clusters"],
                "ignore_values_threshold": parameters["ignore_values_threshold_for_event_attributes"]
            }),
            rng=rng)
        trainResult = m.train(e)
        filename = m.save(parameters["file_handle"], parameters)
        writeLog("Generated model saved into file: %s" % filename)
        print(filename)

        if (parameters["test_filename"] != None):
            m = ModelCluster(rng)
            m.load(filename, parameters)
            el = EventLog(parameters, rng, parameters["test_filename"], modelCluster=m)
            result = m.test(el, 1.0, trainResult)
            jsonResult = json.dumps(result)
            filename = get_filename("predict_result",
                                    "%s_%s_%s" % (parameters["file_handle"], m.case_name, m.eventlog.filename), "json")
            with open(filename, "w") as f:
                f.write(jsonResult)
            writeLog("Generated results saved into file: %s" % filename)
            print(jsonResult)


def isFile(filename):
    path = Path(filename)
    try:
        if (path.is_file()):
            return True
    except:
        return False
    return False


def testPaused(parameters):
    wasPaused = False
    while True:
        filename = parameters["pause_filename"]
        if (not isFile(filename)):
            filename = getInputPath() + parameters["pause_filename"]
            if (not isFile(filename)):
                filename = getOutputPath() + parameters["pause_filename"]
                if (not isFile(filename)):
                    break
        if not wasPaused:
            writeLog("Tests paused until file is removed: %s" % filename)
            wasPaused = True
        sleep(1)
    if wasPaused:
        writeLog("Tests continued...")


def waitForConfiguration(origFilename, parameters):
    wasPaused = False
    filename = None
    while True:
        filename = origFilename
        if isFile(filename):
            break

        filename = getInputPath() + origFilename
        if isFile(filename):
            break

        filename = getOutputPath() + origFilename
        if isFile(filename):
            break

        if not wasPaused:
            writeLog("Tests paused until a new configuration file appears in: %s" % origFilename)
            wasPaused = True
        sleep(1)
    if wasPaused:
        writeLog("Got new configuration. Continuing...")
    writeLog("Reading new configuration from %a" % filename)
    result = loadConfiguration(filename, parameters)
    os.remove(filename)
    return result


def collect(configuration, parameters, to):
    if isinstance(configuration, list):
        to += configuration
        return True
    conf = dict(configuration)
    parameters = dict(parameters)
    if (("exit" in parameters) and (parameters["exit"])):
        return False

    result = True
    if "for_each" in conf:
        for iterConf in conf["for_each"]:
            newConf = dict(conf)
            newConf.update(iterConf)
            newConf.pop("for_each")
            result &= collect(newConf, parameters, to)
        return result
    if "include" in conf:
        filename = conf["include"]
        conf.pop("include")
        parameters.update(conf)
        if (not isFile(filename)):
            filename = default_parameters["input_directory"] + conf["include"]
        if (not isFile(filename) and (configurationPath != None)):
            filename = str(configurationPath.parent) + conf["include"]
        with open(filename) as data:
            newConf = json.load(data)
        result &= collect(newConf, parameters, to)
    else:
        leaf = not ("runs" in conf)
        if not leaf:
            conf.pop("runs")
        parameters.update(conf)

        if (leaf):
            if (("skip" in parameters) and (parameters["skip"])):
                return result
            to.append(parameters)
        else:
            for childConfiguration in configuration["runs"]:
                result &= collect(childConfiguration, parameters, to)
    return result


def loadConfiguration(filename, parameters):
    global configurationPath
    path = Path(filename)
    if (not path.is_file()):
        filename = default_parameters["input_directory"] + filename
    configurationPath = Path(filename)
    if (configurationPath.is_file()):
        with open(filename) as data:
            configuration = json.load(data)
        if isinstance(configuration, list):
            configuration = {
                "runs": configuration
            }
        parameters.update(configuration)
        configure(parameters["input_directory"], parameters["output_directory"], opts.log_to_file_only)
        return configuration
    return None


def main(configuration, parameters):
    def saveConfigs(testConfigs):
        jsonConfig = json.dumps(testConfigs)
        with open(started_tests_filename, "w") as f:
            f.write(jsonConfig)

    if configuration != None:
        tests = []
        if Path(started_tests_filename).is_file():
            tsts = None
            with open(started_tests_filename) as data:
                tsts = json.load(data)
            for t in tsts:
                ts = dict(default_parameters)
                ts.update(t)
                tests.append(ts)
            writeLog("Loaded remaining %d test configurations from %s." % (len(tests), started_tests_filename))
        else:
            if (not collect(configuration, default_parameters, tests)):
                writeLog("Exit requested. Finishing tests...")
                return

            saveConfigs(tests)
            writeLog("Generated %d test configurations." % (len(tests)))

        if opts.skip_tests > 0:
            tests = tests[opts.skip_tests:]
            saveConfigs(tests)
            writeLog("Skipping %d first test configurations leaving total of %d test remaining." % (
            opts.skip_tests, len(tests)))

        testPaused(parameters)
        nTests = len(tests)
        i = 1
        while (len(tests) > 0):
            writeLog("Starting test %d of %d." % (i, nTests))
            try:
                run(tests[0])
            except:
                writeLog("Exception: " + traceback.format_exc())
            tests = tests[1:]
            saveConfigs(tests)
            testPaused(parameters)
            i = i + 1

        os.remove(started_tests_filename)

    if ("test_config_filename" in default_parameters) and (default_parameters["test_config_filename"] != None):
        parameters = dict(default_parameters)
        configuration = waitForConfiguration(parameters["test_config_filename"], parameters)
        main(configuration, parameters)
    writeLog("Tests finished.")


started_tests_filename = default_parameters["output_directory"] + "current-tests.json"

if (opts.configuration_from_standard_input):
    writeLog("Reading configuration from standard input")
    jsonConfig = sys.stdin.readline()
    configuration = json.loads(jsonConfig)
    writeLog("Standard input reading finished")

parameters = dict(default_parameters)
configuration = None
if (opts.configuration_filename != None):
    configuration = loadConfiguration(opts.configuration_filename, parameters)
configure(parameters["input_directory"], parameters["output_directory"], opts.log_to_file_only)

writeLog(__doc__)

if __name__ == '__main__':
    main(configuration, parameters)
