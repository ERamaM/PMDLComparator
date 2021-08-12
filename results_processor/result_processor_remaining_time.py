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

dir_to_approach = {
    "ImagePPMiner" : "pasquadibisceglie",
    "nnpm" : "mauro",
    "PyDREAM-NAP" : "theis",
    "GenerativeLSTM" : "camargo",
    "MAED-TaxIntegration" : "khan",
    "DALSTM" : "navarin"
}
directories = [
    "../tax/code/results",
    "../GenerativeLSTM/output_files/",
    "../DALSTM/results/"
]

# These regexes allow us to find the file that contains the results
file_approaches_regex = {
    "tax": "suffix_.*.csv",  # Ends with "next_event.log"
    "navarin" : ".*\.txt$"
}

# These regexes allow us to find the line inside the result file that contains the accuracy
approaches_accuracy_regexes = {
    "tax": "Mean MAE: (.*)",
    "navarin" : "Root Mean Squared Error: \d+\.\d+ d MAE: (\d+\.\d+) d MAPE: .*"
}

# These regexes allow us to delete parts of the filename that are not relevant
approaches_clean_log_regexes = {
    "tax": [".csv", "suffix_"],
    "navarin" : [".csv.txt"]
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
                clean_filename = file
                for to_clean in approaches_clean_log_regexes[approach]:
                    clean_filename = clean_filename.replace(to_clean, "")
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
    relevant_rows = csv[["implementation", "accuracy", "file_name"]][csv["implementation"] == "Arg Max"]
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
# Perform paired t-test and retrieve results
############################################
"""
t_results = {}
p_results = {}
for approach_A, approach_B in itertools.permutations(results.keys(), 2):
    for log in list(available_logs):
        s2_list = []
        # We need to sort to guarantee that the first element is always the same (e.g., fold 0 vs fold 0 and so on)
        # Calculation from here: http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/
        p_mean = 0
        first_p = 0
        for fold_A, fold_B in zip(sorted(results[approach_A][log].keys()), sorted(results[approach_B][log].keys())):
            # Error estimates
            p1 = results[approach_A][log][fold_A]["0"] - results[approach_B][log][fold_B]["0"]
            p2 = results[approach_A][log][fold_A]["1"] - results[approach_B][log][fold_B]["1"]
            p_mean = (p1 + p2) / 2
            # We need this information to calculate the t-statistic
            if fold_A == "0":
                first_p = p1

            s2 = (p1 - p_mean) ** 2 + (p2 - p_mean) ** 2  # Variance of the ith replication
            s2_list.append(s2)

        t_statistic = first_p / math.sqrt((1 / len(s2_list)) * sum(s2_list))
        # For some reason you must multiply by 2
        # I think it is because you are performing a two-sided test
        # https://github.com/rasbt/mlxtend/blob/7c329e40fab7659c3dd9b77f31a55db6f2400521/mlxtend/evaluate/ttest.py#L335
        p_value = t.sf(t_statistic, len(s2_list)) * 2
        print("T statistic: ", t_statistic)
        print("P value: ", p_value)
        approach_pair = (approach_A.capitalize(), approach_B.capitalize())
        log_cap = " ".join([x.capitalize() for x in log.split("_")])
        if approach_pair not in t_results:
            t_results[approach_pair] = {}
        t_results[approach_pair][log_cap] = t_statistic
        if approach_pair not in p_results:
            p_results[approach_pair] = {}
        p_results[approach_pair][log_cap] = p_value

print("T results: ", t_results)
print("Available logs: ", available_logs)

def bold_p_value_formatter(x):
    if x < 0.05:
        return r"\textbf{" + str(round(x, 4)) + "}"
    else:
        return str(round(x, 4))

p_df = pd.DataFrame(p_results).transpose()
print("P value df")
print(p_df)
p_latex = p_df.to_latex(formatters=[bold_p_value_formatter] * len(available_logs), escape=False, caption="P-values for the pairwise comparison approaches")
print(p_latex)

t_df = pd.DataFrame.from_dict(t_results).transpose()
print("T statistic df")
print(t_df)
t_latex = t_df.to_latex(escape=False, caption="T value statistic for the pairwise comparison approaches")
print(t_latex)
"""

############################################
# Retrieve average accuracy results from cross-validation to build accuracy matrix
############################################
accuracy_results = {}
accuracy_fold_results = []
for approach in results.keys():
    for log in available_logs:
        log_values = []
        if approach == "camargo" and (log == "nasa" or log == "sepsis"):
            continue
        for fold in results[approach][log].keys():
            log_values.append(results[approach][log][fold]["0"])
            #log_values.append(results[approach][log][fold]["1"])


        mean_val = statistics.mean(log_values)
        log_cap = " ".join([x.capitalize() for x in log.split("_")])
        for fold in results[approach][log].keys():
            accuracy_fold_results.append({"approach" : approach.capitalize(), "log" : log, "fold" : fold, "acc" : results[approach][log][fold]["0"]})
        if not approach.capitalize() in accuracy_results:
            accuracy_results[approach.capitalize()] = {}
        accuracy_results[approach.capitalize()][log_cap] = mean_val

acc_df = pd.DataFrame.from_dict(accuracy_results, orient="index")
acc_df.sort_index(axis=1, inplace=True)
acc_df.sort_index(axis=0, inplace=True)
print("Accuracy df")
print(acc_df)
acc_fold_df = pd.DataFrame.from_dict(accuracy_fold_results)
acc_fold_df.sort_index(axis=1, inplace=True)
acc_fold_df.sort_index(axis=0, inplace=True)
print("Accuracy fold df")
print(acc_fold_df)

acc_df_latex = acc_df.copy()
print("ACC DF LATX: ", acc_df_latex)

# Format to select the best three approaches and assign them colors
for column in acc_df_latex.columns:
    acc_df_latex[column] = acc_df_latex[column].round(3)
    best_three = acc_df_latex[column].nlargest(3)
    acc_df_latex[column] = acc_df_latex[column].astype(str)
    colors = ["PineGreen", "orange", "red"]
    for approach, color in zip(best_three.index, colors):
        acc_df_latex[column].loc[approach] = r"\textcolor{" + color + r"}{\textbf{" + acc_df_latex[column].loc[approach] + "}}"

acc_latex = acc_df_latex.to_latex(escape=False, caption="Mean accuracy of the 10-fold 5x2cv")
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

os.makedirs("./processed_results/csv/remaining_time", exist_ok=True)
os.makedirs("./processed_results/latex/remaining_time", exist_ok=True)


# Save csvs
pairwise_scores.round(4).to_csv("./processed_results/csv/remaining_time/friedman_nemenyi_posthoc.csv")
ranks.round(4).to_csv("./processed_results/csv/remaining_time/raw_ranks.csv")
avg_rank.round(4).to_csv("./processed_results/csv/remaining_time/avg_rank.csv")
acc_df.to_csv("./processed_results/csv/remaining_time/results.csv")
acc_fold_df.to_csv("./processed_results/csv/remaining_time/raw_results.csv")

#p_df.to_csv("./processed_results/csv/p_values_t_test.csv")
#t_df.round(4).to_csv("./processed_results/csv/t_statistic_t_test.csv")

# Save latex
with open("./processed_results/latex/remaining_time/acc_latex.txt", "w") as f:
    f.write(acc_latex)
#with open("../processed_results/latex/p_latex.txt", "w") as f:
#    f.write(p_latex)
#with open("../processed_results/latex/t_latex.txt", "w") as f:
#    f.write(t_latex)