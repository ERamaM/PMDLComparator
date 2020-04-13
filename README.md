# Installation

Use anaconda to install some dependencies and activate the environment:

    conda create -n "tf_2.0_ppm" python=3.6 tensorflow=2.1.0
    conda activate tf_2.0_ppm
    
If you want to use a GPU (recommended) install this instead:;

    conda create -n "tf_2.0_ppm" python=3.6 tensorflow-gpu=2.1.0
    conda activate tf_2.0_ppm


Install additional dependencies:

    python -m pip install pm4py==1.2.12 hyperopt==0.2.3
    
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

