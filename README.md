# Installation

Use anaconda to install some dependencies and activate the environment:

    conda create -n "tf_2.0_ppm" python=3.6 tensorflow=2.1.0
    conda activate tf_2.0_ppm
    
If you want to use a GPU (recommended) install this instead:;

    conda create -n "tf_2.0_ppm" python=3.6 tensorflow-gpu=2.1.0
    conda activate tf_2.0_ppm


Install additional dependencies:

    python -m pip install pm4py==1.2.12 hyperopt==0.2.3 jellyfish==0.7.2 distance==0.1.3 strsim==0.0.3 pyyaml==5.3.1 nltk==3.5 swifter==0.304
    
If you want to use the task spooler, install, in debian based systems:

    apt install task-spool
    
For every approach, a bash script named "generate_queue.sh" is provided that adds a series of jobs to train the neural networks in the datasets.
    
# Prepare the datasets

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

Examples:

    python dataset_processor.py --net mauro --dataset raw_datasets/Helpdesk.xes.gz
    
    python dataset_processor.py --net mauro --batch raw_datasets
    
# Run the experiments

## Pasquadibisceglie (ImagePPMiner)

Run the experiments with the following command:

    python run.py --dataset dataset/LOG_CSV_FILE
    
Where LOG_CSV_FILE are one log inside the "dataset" folder. 

The results of the testing are outputted inside the "results" folder.

The best model is inside the "models" folder.

# Mauro (nnpm)

Run the experiments with the following command

    python deeppm_act.py data/Helpdesk.csv ACT results/helpdesk_results.log
    
Do NOT run the experiments with "BOTH" instead of ACT since the neural network is not programmed for that.

# Tax

Run the training procedure and next event prediction with the following command (inside the "code" folder)

    python train.py --dataset ../data/Helpdesk.csv