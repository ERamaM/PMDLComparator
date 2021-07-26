import os
import re
import itertools
import math
from scipy.stats import t
import pandas as pd
import statistics
from scipy.stats import friedmanchisquare
import numpy as np
import scikit_posthocs as posthocs

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--metric", required=True, choices=["accuracy", "mcc", "f1-score", "precision", "recall", "brier"])
arguments = parser.parse_args()
metric = arguments.metric

dir_to_approach = {
    "ImagePPMiner" : "pasquadibisceglie",
    "nnpm" : "mauro",
    "PyDREAM-NAP" : "theis",
    "GenerativeLSTM" : "camargo",
    "MAED-TaxIntegration" : "khan"
}
directories = [
    "../tax/code/results",
    "../evermann/results",
    "../ImagePPMiner/results",
    "../nnpm/results",
    "../hinkka/src/output",
    "../PyDREAM-NAP/results",
    "../PyDREAM-NAP/results_no_resources",
    "../GenerativeLSTM/output_files/",
    "../MAED-TaxIntegration/busi_task/data"
]

# These regexes allow us to find the file that contains the results
file_approaches_regex = {
    "tax": ".*next_event.log",  # Ends with "next_event.log"
    "evermann": ".*\.txt$",  # Does not start with "raw" # TODO: what about suffix calculations
    "pasquadibisceglie" : "^(?!raw).*",
    "mauro" : "^fold.*.txt",
    "hinkka" : "results_.*",
    "theis": ".*\.txt$",
    "khan" : "results_.*"
}

# These regexes allow us to find the line inside the result file that contains the accuracy
if metric == "accuracy":
    approaches_accuracy_regexes = {
        "tax": "ACC Sklearn: (.*)",
        "evermann": "Accuracy: (.*)",
        "pasquadibisceglie" : "Accuracy: (.*)",
        "mauro" : "Final Accuracy score:.*\[(.*)\]",
        "hinkka": "Accuracy sklearn: (.*)",
        "theis" : "    \"test_acc\": (.*),",
        "khan" : "Accuracy: (.*)"
    }
elif metric == "mcc":
    approaches_accuracy_regexes = {
        "evermann" : "MCC: (.*)",
        "hinkka" : "MCC: (.*)",
        "pasquadibisceglie" : "MCC: (.*)",
        "mauro" : "MCC: (.*)",
        "theis": "    \"test_mcc\": (.*),",
        "tax"  : "MCC: (.*)",
        "khan" : "" # TODO
    }
elif metric == "f1-score":
    approaches_accuracy_regexes = {
        "evermann" : "Weighted f1: (.*)",
        "hinkka" : "Weighted f1: (.*)",
        "pasquadibisceglie" : "Weighted F1: (.*)",
        "mauro" : "Weighted F1: (.*)",
        "theis": "    \"test_fscore_weighted\": (.*),",
        "tax" : "Weighted F1: (.*)",
        "khan": ""  # TODO
    }
elif metric == "precision":
    approaches_accuracy_regexes = {
        "evermann" : "Weighted precision: (.*)",
        "hinkka" : "Weighted Precision: (.*)",
        "pasquadibisceglie" : "Weighted Precision: (.*)",
        "mauro" : "Weighted Precision: (.*)",
        "theis": "    \"test_prec_weighted\": (.*),",
        "tax" : "Weighted Precision: (.*)",
        "khan": ""  # TODO
    }
elif metric == "recall":
    approaches_accuracy_regexes = {
        "evermann" : "Weighted recall: (.*)",
        "hinkka" : "Weighted recall: (.*)",
        "pasquadibisceglie" : "Weighted Recall: (.*)",
        "mauro" : "Weighted Recall: (.*)",
        "theis": "    \"test_rec_weighted\": (.*),",
        "tax" : "Weighted Recall: (.*)",
        "khan": ""  # TODO
    }
elif metric == "brier":
    approaches_accuracy_regexes = {
        "evermann": "Brier score: (.*)",
        "hinkka": "Brier score: (.*)",
        "pasquadibisceglie": "Brier score: (.*)",
        "mauro" : "Brier score: (.*)",
        "theis": "    \"test_brier_score\": (.*),",
        "tax" : "Brier score: (.*)",
        "khan": ""  # TODO
    }
else:
    raise ValueError

# These regexes allow us to delete parts of the filename that are not relevant
approaches_clean_log_regexes = {
    "tax": "_next_event.log",
    "evermann": ".xes.txt",
    "pasquadibisceglie" : ".txt",
    "mauro" : ".txt",
    "hinkka" : "results_",
    "theis" : ".xes.gz_results.txt",
    "khan" : "results_"
}

approaches_by_csv = ["camargo"]

log_regex = "fold(\\d)_variation(\\d)_(.*)"

# Structure: approach -> log -> fold -> variation -> result
results = {}

def store_results(result_dict, approach, log, fold, variation, result):
    # Store results
    if approach not in results:
        result_dict[approach] = {}
    if log not in results[approach]:
        result_dict[approach][log] = {}
    if fold not in results[approach][log]:
        result_dict[approach][log][fold] = {}
    if variation not in results[approach][log][fold]:
        result_dict[approach][log][fold][variation] = result

def extract_by_regex(directory, approach, results, real_approach=None):
    if real_approach is not None:
        results[real_approach] = {}
    else:
        results[approach] = {}
    regex = file_approaches_regex[approach]
    accuracy_regex = approaches_accuracy_regexes[approach]
    for file in os.listdir(directory):
        z = re.match(regex, file)
        if z is not None:

            with open(os.path.join(directory, file), "r") as result_file:
                lines = result_file.readlines()
                accuracy_line = ""
                for line in lines:
                    acc_regex_match = re.match(accuracy_regex, line)
                    if acc_regex_match is not None:
                        accuracy = float(acc_regex_match.group(1))
                        break
                clean_filename = file.replace(approaches_clean_log_regexes[approach], "")
                parse_groups = re.match(log_regex, clean_filename)
                fold, variation, log = parse_groups.groups()
                log = log.lower()
                available_logs.add(log)
                if real_approach is not None:
                    store_results(results, real_approach, log, fold, variation, accuracy)
                else:
                    store_results(results, approach, log, fold, variation, accuracy)

def extract_by_csv_camargo(directory, approach, results):
    if not os.path.exists(os.path.join(directory, "ac_predict_next.csv")):
        return
    csv = pd.read_csv(os.path.join(directory, "ac_predict_next.csv"))

    if metric == "accuracy":
        get_metric = "accuracy"
    elif metric == "mcc":
        get_metric = "mcc"
    elif metric == "brier":
        get_metric = "brier_score"
    elif metric == "f1-score":
        get_metric = "f1_score_weighted"
    elif metric == "precision":
        get_metric = "precision_weighted"
    elif metric == "recall":
        get_metric = "recall_weighted"
    else:
        raise ValueError

    relevant_rows = csv[["implementation", get_metric, "file_name"]][csv["implementation"] == "Arg Max"]
    for idx, row in relevant_rows.iterrows():
        clean_filename = row["file_name"].replace(".csv", "")
        parse_groups = re.match(log_regex, clean_filename)
        fold, variation, log = parse_groups.groups()
        log = log.lower()
        available_logs.add(log)
        store_results(results, approach, log, fold, variation, row["accuracy"])


############################################
# Parse the result files for accuracy information
############################################
available_logs = set()
for directory in directories:
    approach = directory.split("/")[1]
    if approach in dir_to_approach.keys():
        approach = dir_to_approach[approach]
    if approach == "camargo":
        extract_by_csv_camargo(directory, approach, results)
    elif approach == "theis":
        type_exp = directory.split("/")[-1]
        if "resource" not in type_exp:
            real_approach = "Theis_resource"
            extract_by_regex(directory, approach, results, real_approach=real_approach)
        else:
            real_approach = "Theis_no_resource"
            extract_by_regex(directory, approach, results, real_approach=real_approach)
    else:
        extract_by_regex(directory, approach, results)

for approach in results.keys():
    print("Results: ", approach, ":", results[approach])
print("Available logs: ", available_logs)

# Check for empty approaches
delete_list = []
for approach in results.keys():
    if not results[approach]:
        delete_list.append(approach)
for delete in delete_list:
    results.pop(delete)

############################################
# Retrieve average accuracy results from cross-validation to build accuracy matrix
############################################
accuracy_results = {}
accuracy_fold_results = []
for approach in results.keys():
    for log in available_logs:
        if approach == "camargo" and (log == "nasa" or log == "sepsis"):
            continue
        log_values = []
        for fold in results[approach][log].keys():
            log_values.append(results[approach][log][fold]["0"])
            #log_values.append(results[approach][log][fold]["1"])


        mean_val = statistics.mean(log_values)
        log_cap = " ".join([x.capitalize() for x in log.split("_")])
        for fold in results[approach][log].keys():
            accuracy_fold_results.append({"approach" : approach.capitalize(), "log" : log, "fold" : fold, metric : results[approach][log][fold]["0"]})
        if not approach.capitalize() in accuracy_results:
            accuracy_results[approach.capitalize()] = {}
        accuracy_results[approach.capitalize()][log_cap] = mean_val

acc_df = pd.DataFrame.from_dict(accuracy_results, orient="index")
acc_df.sort_index(axis=1, inplace=True)
acc_df.sort_index(axis=0, inplace=True)
print(metric + " df")
print(acc_df)
acc_fold_df = pd.DataFrame.from_dict(accuracy_fold_results)
acc_fold_df.sort_index(axis=1, inplace=True)
acc_fold_df.sort_index(axis=0, inplace=True)
print(metric + " fold df")
print(acc_fold_df)

acc_df_latex = acc_df.copy()
print(metric + " DF LATX: ", acc_df_latex)

# Format to select the best three approaches and assign them colors
for column in acc_df_latex.columns:
    if metric == "accuracy":
        acc_df_latex[column] = acc_df_latex[column] * 100
        acc_df_latex[column] = acc_df_latex[column].round(2)
    else:
        acc_df_latex[column] = acc_df_latex[column].round(4)
    best_three = acc_df_latex[column].nlargest(3)
    acc_df_latex[column] = acc_df_latex[column].astype(str)
    colors = ["PineGreen", "orange", "red"]
    for approach, color in zip(best_three.index, colors):
        acc_df_latex[column].loc[approach] = r"\textcolor{" + color + r"}{\textbf{" + acc_df_latex[column].loc[approach] + "}}"

acc_latex = acc_df_latex.to_latex(escape=False, caption="Mean " + metric + " of the 10-fold 5x2cv")
print(acc_latex)

############################################
# Perform friedman test
############################################
acc_df = acc_df.transpose()
data = np.asarray(acc_df)
stat, p = friedmanchisquare(*data)

alpha = 0.05
reject = p <= alpha
print("Should we reject H0 (i.e. is there a difference in the means) at the", (1-alpha)*100, "% confidence level?", reject)
pairwise_scores = posthocs.posthoc_nemenyi_friedman(acc_df, acc_df.columns)
# WARNING
# For accuracy problems we need to rank in descending order
# STAC does it in ascending order even though the p-values are exactly the same
ranks = acc_df.rank(axis=1, ascending=False)
print("Nemenyi scores: ")
print(pairwise_scores)
avg_rank = ranks.mean()

############################################
# Save results
############################################

os.makedirs("./processed_results/csv/next_activity", exist_ok=True)
os.makedirs("./processed_results/latex/next_activity/plots", exist_ok=True)


# Save csvs
pairwise_scores.round(4).to_csv("./processed_results/csv/next_activity/" + metric + "_friedman_nemenyi_posthoc.csv")
ranks.round(4).to_csv("./processed_results/csv/next_activity/" + metric + "_raw_ranks.csv")
avg_rank.round(4).to_csv("./processed_results/csv/next_activity/" + metric + "_avg_rank.csv")
if metric == "accuracy":
    ((acc_df * 100).round(2)).to_csv("./processed_results/csv/next_activity/" + metric + "_results.csv")
else:
    acc_df.round(4).to_csv("./processed_results/csv/next_activity/" + metric + "_results.csv")
acc_fold_df.to_csv("./processed_results/csv/next_activity/" + metric + "_raw_results.csv")

#p_df.to_csv("./processed_results/csv/p_values_t_test.csv")
#t_df.round(4).to_csv("./processed_results/csv/t_statistic_t_test.csv")

# Save latex
with open("./processed_results/latex/next_activity/" + metric + "_latex.txt", "w") as f:
    f.write(acc_latex)
#with open("../processed_results/latex/p_latex.txt", "w") as f:
#    f.write(p_latex)
#with open("../processed_results/latex/t_latex.txt", "w") as f:
#    f.write(t_latex)