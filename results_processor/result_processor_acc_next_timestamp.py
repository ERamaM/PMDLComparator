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

metric = "nt_mae"

dir_to_approach = {
    "ImagePPMiner" : "pasquadibisceglie",
    "nnpm" : "mauro",
    "PyDREAM-NAP" : "theis",
    "GenerativeLSTM" : "camargo",
    "MAED-TaxIntegration" : "khan"
}
directories = [
    "../tax/code/results",
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
approaches_accuracy_regexes = {
    "tax": "mae_in_days: (.*)",
    "khan" : "MAE: (.*)"
}

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

approaches_by_csv = []

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

    get_metric = "mae"

    relevant_rows = csv[["implementation", get_metric, "file_name"]][csv["implementation"] == "Arg Max"]
    for idx, row in relevant_rows.iterrows():
        clean_filename = row["file_name"].replace(".csv", "")
        parse_groups = re.match(log_regex, clean_filename)
        fold, variation, log = parse_groups.groups()
        log = log.lower()
        available_logs.add(log)
        store_results(results, approach, log, fold, variation, row[get_metric])


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
accuracy_std_results = {}
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
        std_val = statistics.stdev(log_values)
        log_cap = " ".join([x.capitalize() for x in log.split("_")])
        for fold in results[approach][log].keys():
            accuracy_fold_results.append({"approach" : approach.capitalize(), "log" : log, "fold" : fold, metric : results[approach][log][fold]["0"]})
        if not approach.capitalize() in accuracy_results:
            accuracy_results[approach.capitalize()] = {}
            accuracy_std_results[approach.capitalize()] = {}
        accuracy_results[approach.capitalize()][log_cap] = mean_val
        accuracy_std_results[approach.capitalize()][log_cap] = std_val

acc_df = pd.DataFrame.from_dict(accuracy_results, orient="index")
acc_df_std = pd.DataFrame.from_dict(accuracy_std_results, orient="index")
acc_df.sort_index(axis=1, inplace=True)
acc_df.sort_index(axis=0, inplace=True)
acc_df_std.sort_index(axis=1, inplace=True)
acc_df_std.sort_index(axis=0, inplace=True)
print(acc_df)
acc_fold_df = pd.DataFrame.from_dict(accuracy_fold_results)
acc_fold_df.sort_index(axis=1, inplace=True)
acc_fold_df.sort_index(axis=0, inplace=True)
print(metric + " fold df")
print(acc_fold_df)

acc_df_latex = acc_df.copy()
acc_df_latex_std = acc_df_std.copy()
print(metric + " DF LATX: ", acc_df_latex)

def fix_latex_dataset(latex_dataset, paint=True, add=None, largest=True):
    # Format to select the best three approaches and assign them colors
    for column in latex_dataset.columns:
        if metric == "accuracy":
            latex_dataset[column] = latex_dataset[column] * 100
            latex_dataset[column] = latex_dataset[column].round(2)
        else:
            latex_dataset[column] = latex_dataset[column].round(4)
        if largest:
            best_three = latex_dataset[column].nlargest(3)
        else:
            best_three = latex_dataset[column].nsmallest(3)
        latex_dataset[column] = latex_dataset[column].astype(str)
        colors = ["PineGreen", "orange", "red"]
        if paint:
            for approach, color in zip(best_three.index, colors):
                latex_dataset[column].loc[approach] = r"\textcolor{" + color + r"}{\textbf{" + latex_dataset[column].loc[approach] + "}}"

    latex_dataset.rename(columns=lambda x : "\\rotatebox{90}{" + x + "}", inplace=True)
    if add is not None:
        def applymap(x):
            return add + x
        latex_dataset = latex_dataset.applymap(applymap)
    acc_latex = latex_dataset.to_latex(escape=False, caption="Mean " + metric + " of the 5-fold crossvalidation")
    acc_latex = acc_latex.replace("Challenge ", "").replace("Bpi", "BPI")\
        .replace("\\toprule", "").replace("\\midrule", "").replace("\\bottomrule", "")\
        .replace("Theis_no_resource", "Theis et al. (w/o attributes)")\
        .replace("Theis_resource", "Theis et al. (w/ attributes)")\
        .replace("{table}", "{table*}")\
        .replace("\\\\", "\\\\ \hline")\
        .replace("lllllllllllll", "l|cccccccccccc").replace("nan", "-")

    # Fix dataset first column
    acc_latex = acc_latex\
        .replace("BPI 2012 Complete", "\\shortstack[l]{BPI 2012 \\\\ Complete}")\
        .replace("BPI 2012 W Complete", "\\shortstack[l]{BPI 2012 \\\\ W Complete}")\
        .replace("BPI 2013 Closed Problems", "\\shortstack[l]{BPI 2013 \\\\ Closed Problems}")\
        .replace("BPI 2013 Incidents", "\\shortstack[l]{BPI 2013 \\\\ Incidents}")
    print(acc_latex)
    return acc_latex

if metric != "brier":
    acc_latex = fix_latex_dataset(acc_df_latex)
else:
    acc_latex = fix_latex_dataset(acc_df_latex, largest=False, paint=False)
acc_latex_std = fix_latex_dataset(acc_df_latex_std, paint=False, add="$\pm$")

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

os.makedirs("./processed_results/csv/next_timestamp", exist_ok=True)
os.makedirs("./processed_results/latex/next_timestamp/plots", exist_ok=True)
os.makedirs("./processed_results/latex/next_timestamp/plots/delete_camargo", exist_ok=True)
os.makedirs("./processed_results/latex/next_timestamp/plots/delete_sepsis", exist_ok=True)
os.makedirs("./processed_results/csv/next_timestamp/delete_camargo", exist_ok=True)
os.makedirs("./processed_results/csv/next_timestamp/delete_sepsis", exist_ok=True)


# Save csvs
pairwise_scores.round(4).to_csv("./processed_results/csv/next_timestamp/" + metric + "_friedman_nemenyi_posthoc.csv")
ranks.round(4).to_csv("./processed_results/csv/next_timestamp/" + metric + "_raw_ranks.csv")
avg_rank.round(4).to_csv("./processed_results/csv/next_timestamp/" + metric + "_avg_rank.csv")
if metric == "accuracy":
    ((acc_df * 100).round(2)).to_csv("./processed_results/csv/next_timestamp/" + metric + "_results.csv")
else:
    acc_df.round(4).to_csv("./processed_results/csv/next_timestamp/" + metric + "_results.csv")
acc_fold_df.to_csv("./processed_results/csv/next_timestamp/" + metric + "_raw_results.csv")

#p_df.to_csv("./processed_results/csv/p_values_t_test.csv")
#t_df.round(4).to_csv("./processed_results/csv/t_statistic_t_test.csv")

# Save latex
with open("./processed_results/latex/next_timestamp/" + metric + "_latex.txt", "w") as f:
    f.write(acc_latex)
with open("./processed_results/latex/next_timestamp/" + metric + "_std_latex.txt", "w") as f:
    f.write(acc_latex_std)
#with open("../processed_results/latex/p_latex.txt", "w") as f:
#    f.write(p_latex)
#with open("../processed_results/latex/t_latex.txt", "w") as f:
#    f.write(t_latex)