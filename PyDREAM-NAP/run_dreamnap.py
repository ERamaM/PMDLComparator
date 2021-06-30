from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.algo.discovery.heuristics import factory as heuristics_miner
from pm4py.objects.petri.exporter import pnml as pnml_exporter
from DREAM.pydream.EnhancedPN import EnhancedPN
from DREAM.pydream import LogWrapper
import os
import json
import re

from DREAM.pydream.LogWrapper import LogWrapper
from DREAM.pydream.predictive.nap.NAP import NAP
from DREAM.pydream.predictive.nap.NAPr import NAPr

import tensorflow as tf
import numpy as np
import argparse
import yaml
from pathlib import Path

enhanced_pn_folder = "./enhanced_pns/"
best_model_folder = "./best_models/"
# Beware of the last slash. NAP does not use os.path.join
model_checkpont_folder = "./model_checkpoints"
results_folder = "./results"

# Avoid saturating the GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
tf.random.set_seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--fold_dataset", type=str, required=True)
parser.add_argument("--full_dataset", type=str, required=True)
parser.add_argument("--train", help="Start training the neural network", action="store_true")
parser.add_argument("--test", help="Start testing next event of the neural network", action="store_true")
args = parser.parse_args()

dataset_path = args.fold_dataset
dataset_directory = Path(dataset_path).parent
log_name = Path(dataset_path).name

print("Log name: ", log_name)

# Import the yaml and select attributes
with open("attributes.yaml") as yaml_file:
    data = yaml.safe_load(yaml_file)
    if log_name.find(".") != -1:
        name = log_name.split(".")[0]
    else:
        name = log_name

    # Delete the fold information from the name
    if "fold" in name:
        name = re.sub("fold\\d_variation\\d_", "", name)
    attributes = data[name]

if "fold" not in log_name :
    model_regex = "train_val_" + log_name + "_\d\.\d_\d\.\d\.pnml"
else:
    base_name = log_name
    #base_name = re.sub("(fold)(\\d_variation)(\\d_)", "\\1\\2\\3", base_name)
    model_regex = "train_val_" + base_name + "_\d\.\d_\d\.\d\.pnml"
model_regex_logs = "logs_" + model_regex
train_log_file = "./logs/train_" + log_name
val_log_file = "./logs/val_" + log_name
test_log_file = "./logs/test_" + log_name

main_log = xes_import_factory.apply(args.full_dataset)
main_log = LogWrapper(main_log, resources=attributes)

model_file = None
print("Model regex: ", model_regex)
print("Model regex logs: ", model_regex)
for file in os.listdir(best_model_folder):
    if re.match(model_regex, file):
        model_file = file
        break
    if re.match(model_regex_logs, file):
        model_file = file
        break

if model_file is None:
    raise FileNotFoundError("Unable to find mined model. Have you executed the splitminer?. Searching for: " + model_regex)

net, initial_marking, final_marking = pnml_importer.import_net(best_model_folder + model_file)


def load_and_process(log_file, type):
    log = xes_import_factory.apply(log_file)

    log_wrapper = LogWrapper(log, resources=attributes)
    print("Resource keys: ", main_log.getResourceKeys())
    # The number of resources in
    log_wrapper.setResourceKeys(main_log.getResourceKeys())
    enhanced_pn = EnhancedPN(net, initial_marking)
    enhanced_pn.enhance(log_wrapper)

    timedstatesamples_json, tss_objs = enhanced_pn.decay_replay(log_wrapper=log_wrapper, resources=attributes)

    save_json_path = enhanced_pn_folder + type + log_name + "_timedstatesamples.json"

    with open(save_json_path, "w") as f:
        json.dump(timedstatesamples_json, f)

    return save_json_path


# Find the best model corresponding to the log
if not os.path.isdir(enhanced_pn_folder):
    os.mkdir(enhanced_pn_folder)

full_file = load_and_process(args.full_dataset, "")
train_file = load_and_process(train_log_file, "train_")
val_file = load_and_process(val_log_file, "val_")
test_file = load_and_process(test_log_file, "test_")

# If there is no attribute
if attributes is None or not attributes:
    nap = NAP(tss_train_file=train_file, tss_full_log=full_file, tss_val_file=val_file, tss_test_file=test_file, options={"n_epochs": 100})
else:
    nap = NAPr(tss_train_file=train_file, tss_full_log=full_file, tss_val_file=val_file, tss_test_file=test_file, options={"n_epochs": 100})
if not os.path.isdir(model_checkpont_folder):
    os.mkdir(model_checkpont_folder)

if args.train:
    nap.train(checkpoint_path=model_checkpont_folder, name=log_name, save_results=True)
if args.test:
    nap.loadModel(model_checkpont_folder, log_name)
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    nap.perform_test(
        os.path.join(results_folder, log_name + "_results.txt"),
        os.path.join(results_folder, "raw_" + log_name + ".txt")
    )


"""
enhanced_pn.saveToFile(enhanced_pn_folder + log_name + ".json")
"""
