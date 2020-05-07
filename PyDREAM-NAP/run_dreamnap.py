from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.algo.discovery.heuristics import factory as heuristics_miner
from pm4py.objects.petri.exporter import pnml as pnml_exporter
from PyDREAM.pydream.EnhancedPN import EnhancedPN
from PyDREAM.pydream import LogWrapper
import os
import json
import re

from PyDREAM.pydream.LogWrapper import LogWrapper
from PyDREAM.pydream.predictive.nap.NAP import NAP
from PyDREAM.pydream.predictive.nap.NAPr import NAPr



# log_name = "BPI_Challenge_2012_W_Complete.xes.gz"
# log_name = "Helpdesk.xes.gz"
log_name = "bpi_challenge_2013_incidents.xes.gz"
attributes = ["org:group", "resource country", "organization involved", "org:role", "impact", "product", "lifecycle:transition"]
# attributes = ["org:resource"]

enhanced_pn_folder = "./enhanced_pns/"
best_model_folder = "./best_models/"
model_file = None

model_regex = "train_val_" + log_name + "_\d\.\d_\d\.\d\.pnml"
train_log_file = "./logs/train_val_" + log_name
test_log_file = "./logs/test_" + log_name

total_log_file = "./logs/" + log_name

main_log = xes_import_factory.apply(total_log_file)
main_log = LogWrapper(main_log, resources=attributes)


def load_and_process(log_file, regex, type):
    for file in os.listdir(best_model_folder):
        if re.match(regex, file):
            model_file = file
            break
    log = xes_import_factory.apply(log_file)

    net, initial_marking, final_marking = pnml_importer.import_net(best_model_folder + model_file)

    """
    if not type == "test_":
        net, im, fm = heuristics_miner.apply(log, parameters={"dependency_thresh": 0.99})
        pnml_exporter.export_net(net, im, "discovered_pn.pnml")

    net, initial_marking, final_marking = pnml_importer.import_net("discovered_pn.pnml")
    """

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

train_file = load_and_process(train_log_file, model_regex, "train_")
test_file = load_and_process(test_log_file, model_regex, "test_")


nap = NAPr(tss_train_file=train_file, tss_test_file=test_file, options={"n_epochs" : 100})
if not os.path.isdir("checkpoints"):
    os.mkdir("checkpoints")
nap.train(checkpoint_path="checkpoints", name="NAP", save_results=True)


"""

enhanced_pn.saveToFile(enhanced_pn_folder + log_name + ".json")
"""
