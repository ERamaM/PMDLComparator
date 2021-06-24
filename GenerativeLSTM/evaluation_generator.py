# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:08:16 2019

@author: Manuel Camargo
"""
import datetime

def create_file_list(path):
    file_list = list()
    for root, dirs, files in os.walk(path):
        for f in files:
            file_list.append(f)
    return file_list


def create_folder_list(path, num_models):
    file_list = list()
    for _, dirs, _ in os.walk(path):
        for d in dirs:
            for _, _, files in os.walk(os.path.join(path, d)):
                files_filtered = list()
                for f in files:
                    _, file_extension = os.path.splitext(f)
                    if file_extension == '.h5':
                        files_filtered.append(f)
                creation_list = list()
                for f in files_filtered:
                    date = os.path.getmtime(os.path.join(path, d, f))
                    creation_list.append(
                        {'filename': f,
                         'creation': datetime.datetime.utcfromtimestamp(date)})
                creation_list = sorted(creation_list,
                                       key=lambda x: x['creation'],
                                       reverse=True)
                for f in creation_list[:num_models]:
                    file_list.append(dict(folder=d, file=f['filename']))
    return file_list


import argparse, gzip, shutil, os
from pathlib import Path
import pandas as pd
pd.set_option('display.max_colwidth',1000)
parser = argparse.ArgumentParser(description="Generate training script")
parser.add_argument("--log", help="Log to generate the script", required=True)
args = parser.parse_args()
log_path = args.log
log_name = Path(log_path).name
log_dir = Path(log_path).parent
import shutil
if shutil.which("tsp") is not None:
    tsp_executable = "tsp"
else:
    tsp_executable = "ts"

loss_file = os.path.join("output_files", log_name, "losses_" + log_name)
loss_df = pd.read_csv(loss_file, sep=";", index_col=False)
min_loss = loss_df[loss_df.loss == loss_df.loss.min()]
best_model = min_loss["best_model"].to_string()

split = best_model.split("/")
for s in split:
    print("split: ", s)
folder = os.path.join(split[1], split[2])
model = os.path.join(split[-1])


command_next = tsp_executable + " python lstm.py -a predict_next -c " + folder + " -b \"" + model + "\" -x False -ho True"
command_sfx = tsp_executable + " python lstm.py -a pred_sfx -c " + folder + " -b \"" + model + "\" -x False -ho True -t 100"
os.system("TS_SOCKET=/tmp/camargo " + command_next)
os.system("TS_SOCKET=/tmp/camargo " + command_sfx)

