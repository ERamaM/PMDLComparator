# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:08:16 2019

@author: Manuel Camargo
"""
import itertools
from support_modules import support as sup
import os
import random
random.seed(42) # The seed makes the set of explored hyperparameters the same
import time

# =============================================================================
#  Support
# =============================================================================



def create_file_list(path):
    file_list = list()
    for root, dirs, files in os.walk(path):
        for f in files:
            file_list.append(f)
    return file_list

# =============================================================================
# Experiments definition
# =============================================================================


def configs_creation(num_choice=0):
    configs = list()
    if is_single_exp:
        config = dict(
            lstm_act='relu',
            dense_act=None,
            optimizers='Adam',
            norm_method='lognorm',
            n_sizes=15,
            l_sizes=100)
        for model in model_type:
            configs.append({**{'model_type': model}, **config})
    else:
        # Search space definition
        dense_act = ['sigmoid', 'linear', None]
        norm_method = ['max']
        l_sizes = [50, 100, 200]
        optimizers = ['Nadam', 'Adam']
        lstm_act = ['tanh', 'sigmoid', 'relu']
        if arch == 'sh':
            n_sizes = [5, 10, 15]
            listOLists = [lstm_act, dense_act,
                          norm_method, n_sizes,
                          l_sizes, optimizers]
        else:
            listOLists = [lstm_act, dense_act,
                          norm_method, l_sizes,
                          optimizers]
        # selection method definition
        choice = 'random'
        preconfigs = list()
        for lists in itertools.product(*listOLists):
            if arch == 'sh':
                preconfigs.append(dict(lstm_act=lists[0],
                                       dense_act=lists[1],
                                       norm_method=lists[2],
                                       n_sizes=lists[3],
                                       l_sizes=lists[4],
                                       optimizers=lists[5]))
            else:
                preconfigs.append(dict(lstm_act=lists[0],
                                       dense_act=lists[1],
                                       norm_method=lists[2],
                                       l_sizes=lists[3],
                                       optimizers=lists[4]))
        # configurations definition
        if choice == 'random':
            preconfigs = random.sample(preconfigs, num_choice)
        for preconfig in preconfigs:
            for model in model_type:
                config = {'model_type': model}
                config = {**config, **preconfig}
                configs.append(config)
    return configs

# =============================================================================
# Sbatch files creator
# =============================================================================


def sbatch_creator(configs):
    for i, _ in enumerate(configs):
        if configs[i]['model_type'] in ['shared_cat', 'seq2seq']:
            exp_name = (os.path.splitext(log)[0]
                        .lower()
                        .split(' ')[0][:4] + arch)
        elif configs[i]['model_type'] in ['shared_cat_inter', 'seq2seq_inter']:
            exp_name = (os.path.splitext(log)[0]
                        .lower()
                        .split(' ')[0][:4] + arch + 'i')
        if imp == 2:
            default = ['#!/bin/bash',
                       '#SBATCH --partition=gpu',
                       '#SBATCH --gres=gpu:tesla:1',
                       '#SBATCH -J ' + exp_name,
                       '#SBATCH -N 1',
                       '#SBATCH --mem=25000',
                       '#SBATCH -t 72:00:00',
                       'module load  python/3.6.3/virtenv',
                       'source activate lstm_dev_cpu'
                       ]
        else:
            default = ['#!/bin/bash',
                       '#SBATCH --partition=main',
                       '#SBATCH -J ' + exp_name,
                       '#SBATCH -N 1',
                       '#SBATCH --mem=25000',
                       '#SBATCH -t 72:00:00',
                       'module load  python/3.6.3/virtenv',
                       'source activate lstm_dev_cpu'
                       ]

        def format_option(short, parm):
            return (' -'+short+' None'
                    if configs[i][parm] is None
                    else ' -'+short+' '+str(configs[i][parm]))

        options = 'python lstm.py -f ' + log + ' -i ' + str(imp)
        options += ' -a training'
        options += ' -o True'
        options += format_option('l', 'lstm_act')
        options += format_option('y', 'l_sizes')
        options += format_option('d', 'dense_act')
        options += format_option('n', 'norm_method')
        options += format_option('m', 'model_type')
        options += format_option('p', 'optimizers')
        if arch == 'sh':
            options += format_option('z', 'n_sizes')

        default.append(options)
        file_name = sup.folder_id()
        sup.create_text_file(default, os.path.join(output_folder, file_name))

def tsp_creator(configs, tsp="ts"):
    commands = []
    for i, _ in enumerate(configs):
        def format_option(short, parm):
            return (' -'+short+' None'
                    if configs[i][parm] is None
                    else ' -'+short+' '+str(configs[i][parm]))

        options = 'python lstm.py -f ' + log + ' -i ' + str(imp)
        options += ' -a training'
        options += ' -o True'
        options += format_option('l', 'lstm_act')
        options += format_option('y', 'l_sizes')
        options += format_option('d', 'dense_act')
        options += format_option('n', 'norm_method')
        options += format_option('m', 'model_type')
        options += format_option('p', 'optimizers')
        if arch == 'sh':
            options += format_option('z', 'n_sizes')

        commands.append(options)
    return commands


# =============================================================================
# Sbatch files submission
# =============================================================================


def sbatch_submit(in_batch, bsize=10):
    file_list = create_file_list(output_folder)
    print('Number of experiments:', len(file_list), sep=' ')
    for i, _ in enumerate(file_list):
        if in_batch:
            if (i % bsize) == 0:
                time.sleep(20)
                os.system('sbatch '+os.path.join(output_folder, file_list[i]))
            else:
                os.system('sbatch '+os.path.join(output_folder, file_list[i]))
        else:
            os.system('sbatch '+os.path.join(output_folder, file_list[i]))

# =============================================================================
# Kernel
# =============================================================================


# create output folder
output_folder = 'jobs_files'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# clean folder
for _, _, files in os.walk(output_folder):
    for file in files:
        os.unlink(os.path.join(output_folder, file))

# parameters definition

is_single_exp = False
# s2, sh
arch = 'sh'

import argparse, gzip, shutil, os
from pathlib import Path
parser = argparse.ArgumentParser(description="Generate training script")
parser.add_argument("--log", help="Log to generate the script", required=True)
parser.add_argument("--execute_inplace", help="Do not write the commands to a file. Instead, execute them directly", action="store_true")
args = parser.parse_args()
log_path = args.log
log_name = Path(log_path).name
log_dir = Path(log_path).parent

if ".gz" in log_path:
    log_name = log_name.replace(".gz", "")
    with gzip.open(log_path, "rb") as f_in:
        with open(os.path.join(log_dir, log_name), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

# Anyways, the code is hardcoded to load the files from "input_files"
log = log_name
# We also hardcode the script output location
output_folder_scripts = "training_scripts"
if not os.path.exists(output_folder_scripts):
    os.mkdir(output_folder_scripts)


imp = 1  # keras lstm implementation 1 cpu, 2 gpu

# Same experiment for both models
if arch == 'sh':
    model_type = ['shared_cat']
else:
    model_type = ['seq2seq_inter', 'seq2seq']

# configs definition
configs = configs_creation(num_choice=20)
print("Configs:", configs)
# sbatch creation

# Search for the executable
import shutil
if shutil.which("tsp") is not None:
    tsp_executable = "tsp"
else:
    tsp_executable = "ts"
commands = tsp_creator(configs, tsp=tsp_executable)

# Set the number of concurrent jobs
if not args.execute_inplace:
    commands = ["cd .. && " + tsp_executable + " python lstm.py -a emb_training -f " + log + " -o True"] + commands
    with open(os.path.join(output_folder_scripts, "execute_order_" + log + ".sh"), "w") as f:
        f.write("#!/bin/bash\n")
        for command in commands:
            f.write(tsp_executable + " " + command + "\n")
else:
    # Wait for the training of the embeddings
    emb_command = "python lstm.py -a emb_training -f " + log + " -o True"
    os.system(emb_command)
    # Send all to tsp
    for command in commands:
        os.system(tsp_executable + " -S 5 " + command)

# submission
# sbatch_submit(True)
