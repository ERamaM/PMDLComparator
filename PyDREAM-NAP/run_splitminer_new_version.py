######
# IMPORTANT: uses PM4pY version 2.29
# IMPORTANT 2: youll need a vncserver
# apt install tightvncserver
# vncserver :1001
# export DISPLAY=localhost:1001
######
import warnings
import tqdm
from pm4py import read_bpmn

warnings.simplefilter("ignore")
warnings.simplefilter("ignore", category=DeprecationWarning)
import argparse, subprocess, os, re
import numpy as np
from shutil import copyfile
from pathlib import Path
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import tempfile

parser = argparse.ArgumentParser(description="Run splitminer")
parser.add_argument("--log", help="Log to process", required=True)
parser.add_argument("--best_model", help="Best model store folder", required=True)
parser.add_argument("--output_folder", help="Output folder of the mined models", required=True)
parser.add_argument("--n_threads", help="Number of threads to use", type=int, required=True)
arguments = parser.parse_args()

# Load log and delete useless attributes to use split miner
# Split miner does not fare well with random lifecycle transitions.
# Change every transition to COMPLETE and forget it.
log_file = Path(arguments.log).name
from pm4py.objects.conversion.log import converter as log_converter
log = xes_importer.apply(arguments.log)
pd_log = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
pd_log = pd_log[["case:concept:name", "concept:name", "time:timestamp", "lifecycle:transition"]]
pd_log["lifecycle:transition"] = "complete"
log = log_converter.apply(pd_log, variant=log_converter.Variants.TO_EVENT_LOG)
tmp_file = tempfile.NamedTemporaryFile().name + ".xes"
xes_exporter.apply(pd_log, tmp_file)
os.system("gzip " + tmp_file)
#striped_log = [{ev["conceptfor trace in log for ev in trace]

print("TMP FILE :", tmp_file)

for eta in range(0, 11):
    eta = eta / 10.0
    for epsilon in range(0, 11):
        epsilon = epsilon / 10.0
        print("java -cp ./split_miner_2.0/sm2.jar:./split_miner_2.0/lib/* au.edu.unimelb.services.ServiceProvider SMD " + str(eta) + " " + str(epsilon) + " true true true " + tmp_file + ".gz" + " " + os.path.join(arguments.output_folder, log_file + "_" + str(eta) + "_" + str(epsilon)))

    raise ValueError


model_regex = log_file + "_\d\.\d_\d\.\d\.bpmn"
model_fitnesses = {}
files_to_process = []
for file in os.listdir(arguments.output_folder):
    print("BONO: ", file)
    print("REGEX: ", model_regex)
    if re.match(model_regex, file):
        print("XD: ", file)
        files_to_process.append(file)

model_fitnesses = {}

import os
def process_file(file):
    # Import petri net and calculate fitness
    # This function throws a warning and it is unavoidable
    #print("Running: ", file)
    bpmn_graph = read_bpmn(os.path.join(arguments.output_folder, file))
    net, initial_marking, final_marking = bpmn_converter.apply(bpmn_graph)
    #net, initial_marking, final_marking = pnml_importer.import_net(os.path.join(arguments.output_folder, file))
    log = xes_importer.apply(arguments.log)
    fitness = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
    try:
        fitness = fitness["averageFitness"]
    except:
        # Depending on the version is one or another
        fitness = fitness["average_trace_fitness"]
    return fitness, file


import multiprocessing as mp
print("Calculating fitnesses. Please wait.")
print("CPU count: ", os.cpu_count())
if arguments.n_threads * 4 <= os.cpu_count():
    max_workers = arguments.n_threads * 6
else:
    max_workers = arguments.n_threads
pool = mp.Pool(processes=max_workers)
for fitness, file in tqdm.tqdm(pool.imap_unordered(process_file, files_to_process), total=len(files_to_process)):
    model_fitnesses[file] = fitness

print("Model fitnesses: ", model_fitnesses)
sorted_fitnesses = {k: v for k, v in sorted(model_fitnesses.items(), key=lambda item: item[1], reverse=True)}
best_model = next(iter(sorted_fitnesses))
print("Best model: ", best_model)
print("Best fitness: ", sorted_fitnesses[best_model])
copyfile(os.path.join(arguments.output_folder, best_model), os.path.join(arguments.best_model, best_model))


