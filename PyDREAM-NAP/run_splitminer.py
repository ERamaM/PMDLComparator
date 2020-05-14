import warnings
import tqdm
warnings.simplefilter("ignore")
warnings.simplefilter("ignore", category=DeprecationWarning)
import argparse, subprocess, os, re
import numpy as np
from shutil import copyfile
from pathlib import Path
from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.algo.conformance.alignments import factory as align_factory
import pm4pycvxopt # Fast alignments

parser = argparse.ArgumentParser(description="Run splitminer")
parser.add_argument("--log", help="Log to process", required=True)
parser.add_argument("--best_model", help="Best model store folder", required=True)
parser.add_argument("--output_folder", help="Output folder of the mined models", required=True)
parser.add_argument("--n_threads", help="Number of threads to use", type=int, required=True)
arguments = parser.parse_args()

os.system("java -jar splitminer_cmd-1.0.0-all.jar -l " + arguments.log + " -b " + arguments.best_model + " -m " + arguments.output_folder + " -t " + str(arguments.n_threads))

log_file = Path(arguments.log).name
model_regex = log_file + "_\d\.\d_\d\.\d\.pnml"
model_fitnesses = {}
files_to_process = []
for file in os.listdir(arguments.output_folder):
    if re.match(model_regex, file):
        files_to_process.append(file)

model_fitnesses = {}
pbar = tqdm.tqdm(total=len(files_to_process))
import os
def process_file(file):
    # Import petri net and calculate fitness
    # This function throws a warning and it is unavoidable
    #print("Running: ", file)
    net, initial_marking, final_marking = pnml_importer.import_net(os.path.join(arguments.output_folder, file))
    log = xes_importer.import_log(arguments.log)
    alignments = pm4pycvxopt.align_factory.apply_log(log, net, initial_marking, final_marking)
    trace_fitnesses = [alignment["fitness"] for alignment in alignments]
    fitness = np.mean(trace_fitnesses)
    model_fitnesses[file] = fitness
    pbar.update(1)
    #print("Finished: ", file, " with fitness: ", fitness)

import concurrent.futures
print("Calculating fitnesses. Please wait.")
print("CPU count: ", os.cpu_count())
if arguments.n_threads * 4 <= os.cpu_count():
    max_workers = arguments.n_threads * 4
else:
    max_workers = arguments.n_threads
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for file in files_to_process:
        executor.submit(process_file, file)

print("Model fitnesses: ", model_fitnesses)
sorted_fitnesses = {k: v for k, v in sorted(model_fitnesses.items(), key=lambda item: item[1], reverse=True)}
best_model = next(iter(sorted_fitnesses))
print("Best model: ", best_model)
print("Best fitness: ", sorted_fitnesses[best_model])
copyfile(os.path.join(arguments.output_folder, best_model), os.path.join(arguments.best_model, best_model))


