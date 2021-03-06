# Deep Learning for Predictive Busines Process Monitoring: Review and Benchmark

IMPORTANT NOTE: this repository will not be updated in the near future. Please use the following one: https://gitlab.citius.usc.es/efren.rama/pmdlcompararator

# Introduction

This is the support code of the paper "Deep Learning for Predictive Business Process Monitoring: Review and Benchmark."

<span style="font-size:larger;">**Important**</span>: the repository contains two branches, namely "master" and "crossvalidation".
The first branch contains the results for the first version of the paper, which used a 64/16/20 split, whereas the second branch contains 
the results for the 5-fold crossvalidation procedure. 

I **strongly** recommend using the crossvalidation branch since *(i)* it contains more approaches *(ii)* the approaches have some bugs solved that are not fixed in the master branch.

## Implemented approaches

| Author                   | Paper                                                                  | Original Code repository                                                                                                              |
|--------------------------|------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Tax et al.               | [Link](https://arxiv.org/abs/1612.02130)                               | [Code](https://github.com/verenich/ProcessSequencePrediction)                                                                         |
| Evermann et al.          | [Link](https://arxiv.org/abs/1612.04600)                               | [Code](https://joerg.evermann.ca/docs/rnn_process_data_scripts_results.tar.gz)                                                        |
| Navarin et al.           | [Link](https://arxiv.org/abs/1711.03822)                               | [Code](https://github.com/nickgentoo/DALSTM_PM)                                                                                       |
| Khan et al.              | [Link](https://arxiv.org/abs/1802.00938v1)                             | [Code](https://github.com/thaihungle/MAED/tree/deep-process)                                                                          |
| Theis et al.             | [Link](https://arxiv.org/abs/1903.05084)                               | [Code](https://github.com/Julian-Theis/DREAM-NAP) and [Code_2](https://github.com/Julian-Theis/PyDREAM)                               |
 | Mauro et al.             | [Link](https://openreview.net/forum?id=OxYPkm8nGEq)                    | [Code](https://github.com/nicoladimauro/nnpm)                                                                                         |
 | Pasquadibisceglie et al. | [Link](https://ieeexplore.ieee.org/document/8786066)                   | [Code](https://github.com/vinspdb/ImagePPMiner)                                                                                       |
 | Camargo et al            | [Link](https://kodu.ut.ee/~dumas/pubs/bpm2019lstm.pdf)                 | [Code](https://github.com/AdaptiveBProcess/GenerativeLSTM/)                                                                           |
 | Hinkka et al.            | [Link](https://arxiv.org/abs/1904.06895)                               | [Code](https://github.com/mhinkka/articles/tree/master/Exploiting%20Event%20Log%20Event%20Attributes%20in%20RNN%20Based%20Prediction) |
 | Francescomarino et al.   | [Link](https://link.springer.com/chapter/10.1007/978-3-319-65000-5_15) | [Code](https://github.com/yesanton/Process-Sequence-Prediction-with-A-priori-knowledge)                                               |

## Additional materials

Results of the experimentation are available in [this link](https://nextcloud.citius.usc.es/index.php/s/Ryk7aoSP4JfGXAk)

## Installation

Use anaconda to install some dependencies and activate the environment:

    conda create -n "tf_2.0_ppm" python=3.6 tensorflow=2.1.0
    conda activate tf_2.0_ppm
    
If you want to use a GPU (recommended) install this instead:;

    conda create -n "tf_2.0_ppm" python=3.6 tensorflow-gpu=2.1.0
    conda activate tf_2.0_ppm

Install additional dependencies:

    python -m pip install pm4py==1.2.12 hyperopt==0.2.3 jellyfish==0.7.2 distance==0.1.3 strsim==0.0.3 pyyaml==5.3.1 nltk==3.5 swifter==0.304 py4j==0.10.9
    
If you want to use the task spooler, install, in debian based systems:

    apt install task-spool
    
For every approach, a bash script named "generate_queue.sh" is provided that adds a series of jobs to train the neural networks in the datasets.
    
## Prepare the datasets

There are two modes to prepare the datasets for training and testing: 

- *Single file*: only a single log is processed (--dataset).
- *Batch*: only the logs inside a folder are processed (--batch).

To prepare the datasets you must specify to whom are you going to prepare the datasets for (with the argument --net).

The structure is:

    python dataset_processor.py --net AUTHOR --dataset LOG_XES_FILE
    
or:

    python dataset_processor.py --net AUTHOR --batch FOLDER
    
Where AUTHOR is one of above:

- pascuadibisceglie
- mauro
- evermann
- navarin
- camargo
- hinkka
- thai
- francescomarino
- theis
- tax

Examples:

    python dataset_processor.py --net mauro --dataset raw_datasets/Helpdesk.xes.gz
    
    python dataset_processor.py --net mauro --batch raw_datasets
    
## Run the experiments

### Pasquadibisceglie (ImagePPMiner)

Run the experiments with the following command:

    python run.py --dataset dataset/[DATASET] --train --test
    
Where DATASET is one log inside the "dataset" folder. 

The results of the testing are outputted inside the "results" folder. There are two types of results:

- raw_\[DATASET\]: contains the next activity results for each prefix tested.
- \[DATASET\]: contains the next activity metrics.

The best model is inside the "models" folder.

### Mauro (nnpm)

Run the experiments with the following command:

    python deeppm_act.py --dataset data/[DATASET] --train --test
    
The preprocessed datasets are placed inside the "data" folder. The models are placed inside the "results" folder. The approach also generates the following files:

- \[DATASET\]: it contains multiple information, such as the validation loss of each of the Hyperopt trials and the next activity metrics, such as accuracy or brier score.

### Tax (tax)

Run the training procedure and next event prediction with the following command (inside the "code" folder). Each "--option" indicates the task to perform.

    python train.py --dataset ../data/[DATASET] --train --test --test_suffix
    
The preprocessed datasets are placed inside the "data" folder. The trained models are placed inside './code/models' and the result files are placed inside './code/results'. For each dataset, it generates 4 files:
    
- raw_suffix_and_remaining_time_\[DATASET\]: it contains a list of all ground truth and predicted suffixes used to calculate remaining time and damerau levenshtein distance.
- suffix_\[DATASET\]: it contains the metrics Damerau-Levenshtein distance and Remaining time:
- raw_\[DATASET\]: it contains a list of all ground truth and predicted next activities.
- \[DATASET\]_next_event: it contains the next activity metrics: accuracy, brier score and next timestamp MAE.

### Evermann (evermann)

Run the training and testing procedure as follows:

    python train.py --dataset data/[DATASET] --train --test --test_suffix
    
Where dataset is one of the xes.gz files from the "data" directory (do NOT use the split files: train_\[DATASET\], val_\[DATASET\] or test_\[DATASET\])

The models are stored inside the "models" folder. The results are stored inside the "results" folder. The results are stored as follows:

- \[DATASET\]: contains the metrics for the next activity prediction problem.
- raw_\[DATASET\]: contains the results for the next activity prediction for each prefix tested.
- suffix_results_\[DATASET\]: contains the suffix predictions for each prefix tested.

Additionally, the Damerau Levenshtein metric is not calculated with the previous command. You will need to execute the following command to perform the calculation:

    python train.py --dataset data/[DATASET] --test_suffix_calculus
    
The results are print out in stdout.

### Navarin (DALSTM)

Run the experiments with the following command:

    python LSTM_sequence_mae.py --dataset data/$i --train --test
    
The models are stored inside the "model" folder. The results are stored inside the "results" folder.

### Hinkka (hinkka)

For this study is better to create a new anaconda environment. Create an environment based on tensorflow-gpu (to install the gpu dependencies) and python3.6

    conda create -n hinkka python=3.6 
    conda activate hinkka
    conda install tensorflow-gpu theano pygpu

Activate the environment and install additional pip dependencies:

    python -m pip install nltk==3.5 pillow==7.1.2 pyclustering==0.9.3.1 regex==2020.5.14 sklearn pandas
    
Then, install theano and lasagne (0.2.dev1):

    python -m pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
   
Then, force Theano to use a GPU. For that, edit the file ~/.theanorc (create it if it does not exist) with the following information

    [global]
    device = cuda
    floatX = float32
   
To run the experiments, you need to specify the json configuration file. This json specifies which datasets are going to be tested and with what parameters. The already provided "pmdlcomparator.json" is already configured to test with every dataset available from the paper. The tests are executed with the following command:

    python main.py -c config/pmdlcomparator.json
    
The configuration automatically takes care of the iteration over the datasets used in the paper. Check whether the gpu is being used by searching the following message at runtime (with your installed graphics card):

    Mapped name None to device cuda: GeForce RTX 2080 SUPER (0000:01:00.0)
    
The results are stored in the directory "output". The code generates a bunch of .txt, .csv, and .json that can be safely removed. The .json MIGHT be a caching procedure so, if the code is behaving strangely, just delete these files generated. The script provided "delete_cache.sh" does precisely just that.

The models parameters are stored inside the "testdata" directory. To remove them use the script "delete_models". The script "delete_cache.sh" removes every .txt and .csv created on the root of the approach folder. The results for the next activity are stored in the "output" folder.

### Camargo (GenerativeLSTM)

An easy way to run the experiments requires the task-spooler tool installed. Then, run the hyperparameter optimizacion procedure with the following command:

    python experiment_generator.py --log input_files/[DATASET] --execute_inplace --slots [N_SLOTS]
    
Where \[DATASET\] is one of the already processed datasets and \[N_SLOTS\] an integer greater than 1 (represents the number of concurrent experimentations done).

The previous command first calculates the embedding matrix from the roles and activities. Then, it trains 20 models using a random search procedure and stores each model validation loss in a file. The bash script "generate_queue_train.sh" performs this procedure for each processed dataset.

After the experiments are done (and this is important, since, otherwise, the information about the loss of the trained models would be incomplete), execute the testing procedure with the following command.

    python evaluation_generator.py --log [DATASET]
    
This command loads the model with the lowest validation loss and performs the testing procedures (next activity and suffix). Note that the "input_files" folder is not specified in the command. After the testing is completed, the results are stored in the "output_files" folder. The most important information from this folder is:

- folders \[DATASET\]: each folder contains the trained models from the hyperparameter optimization procedure and a .csv "losses_\[DATASET\]" which records the validation losses for each model.
- ac_pred_sfx.csv: contains the activity suffix metrics for each sampling procedure and dataset. Each dataset can be recognized for the folder in which the best model is stored.
- ac_predict_next.csv: contains the next activity metrics for each sampling procedure. Often, you would be interested only in the "Argmax" metrics.
- tm_pred_sfx.csv: contains the remaining time metrics for each sampling procedure.

### Khan (MAED-TaxIntegration)

Run the experimentation using the "bpi_run.py" script from inside the "busi_task" folder:

    python bpi_run.py --dataset data/[DATASET] --train --test
    
Unlike the other approaches, this approach does not contain a "generate_queue.sh" script since it is more convenient to execute the experiments manually since they take a lot of time to complete.

The results are stored inside the "data" directory under a file named "results_\[DATASET\]". The model checkpoints are inside the "checkpoints_data". The information inside the "log_data" directory also seems important.

### Theis (PyDREAM-NAP)

This approach runs in two phases. First, you must mine the process models from the training+validation event log. Then, you must use the process model to run the training and testing procedure.

To run the mining procedure, execute the following command:

    python run_splitminer.py --log ./logs/"train_val_"$i --output_folder output_models --best_model best_models --n_threads $N_THREADS
    
This command has the following arguments:

- --log: process log to mine the models for.
- --output_folder: folder where every mined process model is going to be stored.
- --best_model: folder where the best process model is going to be stored.
- --n_threads: maximum number of threads to be used by the mining procedure. Recommended: keep this to a lower value than the maximum number of cores of your machine.

After the mining is complete, you have two options to perform the experimentation:

    python run_dreamnap.py --dataset logs/"$i" --train --test
    python run_dreamnap_no_resources.py --dataset logs/"$i" --train --test
    
The first command runs the version that uses resources and the second command runs the version that uses no resources. 

The results are stored inside the "results" folder. The trained models are stored inside the "model_checkpoints" folder.

### Francescomarino (Process-Sequence-Prediction-with-A-priori-knowledge)

First, download the RuM tool from: https://sep.cs.ut.ee/Main/RuM

Then, execute the tool with the following command:

```
java -jar rum-0.5.3.jar
```

Then, in the tab "Discovery" load the "train_val" split of the XES log.

Set the following parameters in the window:

- Templates
	- Existence[A]
	- Response[A,B]
- General parameters:
	- Min Constraint Support: 10%
	- Vacuity detection: enabled.

Save the model using the .decl extension.

Then, in the tab "Conformance checking", load the same log and the previously saved model, and press "Check".

Group the list by "Constraints" and sort the groups by "Fullfillments". Then, select up to 3 rules of each type (response and existence) that have a number of 50% of fullfillments more than activations sorted by the number of fullfillments.

These rules must be added in the file "formulas.yaml" using the same format as indicated in the comments.

Finally, to run the experiments, execute the training procedure as follows:

```
python train.py --dataset DATASET --train --test --test_suffix --test_suffix_calculus
```

# Citation

This work is published in the journal "IEEE Transactions on Services Computing". If you use this code or results in your research projects, we encourage you to please cite our work:

```
@Article{RamaManeiro2021,
  author    = {Efren Rama-Maneiro and Juan Vidal and Manuel Lama},
  journal   = {{IEEE} Transactions on Services Computing},
  title     = {Deep Learning for Predictive Business Process Monitoring: Review and Benchmark},
  year      = {2021},
  pages     = {1--1},
  doi       = {10.1109/tsc.2021.3139807},
  publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
}
```

