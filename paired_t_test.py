import os
import re
import itertools
import math
from scipy.stats import t
import pandas as pd
import statistics

dir_to_approach = {
    "ImagePPMiner" : "pasquadibisceglie",
    "nnpm" : "mauro"
}
directories = [
    "tax/code/results",
    "evermann/results",
    "ImagePPMiner/results",
    "nnpm/results"
]

# These regexes allow us to find the file that contains the results
file_approaches_regex = {
    "tax": ".*next_event.log",  # Ends with "next_event.log"
    "evermann": "^(?!raw).*$",  # Does not start with "raw" # TODO: what about suffix calculations
    "pasquadibisceglie" : "^(?!raw).*",
    "mauro" : "^fold.*.txt"
}

# These regexes allow us to find the line inside the result file that contains the accuracy
approaches_accuracy_regexes = {
    "tax": "ACC Sklearn: (.*)",
    "evermann": "Accuracy: (.*)",
    "pasquadibisceglie" : "Accuracy: (.*)",
    "mauro" : "Final Accuracy score:.*\[(.*)\]"
}

# These regexes allow us to delete parts of the filename that are not relevant
approaches_clean_log_regexes = {
    "tax": "_next_event.log",
    "evermann": ".xes.txt",
    "pasquadibisceglie" : ".txt",
    "mauro" : ".txt"
}

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


############################################
# Parse the result files for accuracy information
############################################
available_logs = set()
for directory in directories:
    approach = directory.split("/")[0]
    if approach in dir_to_approach.keys():
        approach = dir_to_approach[approach]
    results[approach] = {}
    print("Approach: ", approach)
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
                store_results(results, approach, log, fold, variation, accuracy)
    print("Available logs: ", available_logs)

print("Results: ", results)

############################################
# Perform paired t-test and retrieve results
############################################
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
        t_results[approach_pair] = {log_cap : t_statistic}
        p_results[approach_pair] = {log_cap : p_value}

print("T results: ", t_results)
print("Available logs: ", available_logs)

def bold_p_value_formatter(x):
    if x < 0.05:
        return r"\textbf{" + str(round(x, 4)) + "}"
    else:
        return str(round(x, 4))

p_df = pd.DataFrame.from_dict(p_results, orient="index")
print("P value df")
print(p_df)
latex = p_df.to_latex(formatters=[bold_p_value_formatter] * len(available_logs), escape=False, caption="P-values for the pairwise comparison approaches")
print(latex)

t_df = pd.DataFrame.from_dict(t_results, orient="index")
print("T statistic df")
print(t_df)
latex = p_df.to_latex(escape=False, caption="T value statistic for the pairwise comparison approaches")
print(latex)

############################################
# Retrieve accuracy results from cross-validation
############################################
accuracy_results = {}
for approach in results.keys():
    for log in available_logs:
        log_values = []
        for fold in results[approach][log].keys():
            log_values.append(results[approach][log][fold]["0"])
            log_values.append(results[approach][log][fold]["1"])
        mean_val = statistics.mean(log_values)
        log_cap = " ".join([x.capitalize() for x in log.split("_")])
        accuracy_results[approach.capitalize()] = {log_cap : mean_val}

acc_df = pd.DataFrame.from_dict(accuracy_results, orient="index")
print("Accuracy df")
print(acc_df)

acc_df_latex = acc_df.copy()
print("ACC DF LATX: ", acc_df_latex)

# Format to select the best three approaches and assign them colors
for column in acc_df_latex.columns:
    acc_df_latex[column] = acc_df_latex[column] * 100
    acc_df_latex[column] = acc_df_latex[column].round(2)
    best_three = acc_df_latex[column].nlargest(3)
    acc_df_latex[column] = acc_df_latex[column].astype(str)
    colors = ["PineGreen", "orange", "red"]
    for approach, color in zip(best_three.index, colors):
        acc_df_latex[column].loc[approach] = r"\textcolor{" + color + r"}{\textbf{" + acc_df_latex[column].loc[approach] + "}}"

latex = acc_df_latex.to_latex(escape=False, caption="Mean accuracy of the 10-fold 5x2cv")
print(latex)


