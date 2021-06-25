import os
import re
import itertools
import math
from scipy.stats import t

directories = [
    "tax/code/results",
    "evermann/results"
]

# These regexes allow us to find the file that contains the results
file_approaches_regex = {
    "tax": ".*next_event.log",  # Ends with "next_event.log"
    "evermann": "^(?!raw).*$"  # Does not start with "raw"
}

# These regexes allow us to find the line inside the result file that contains the accuracy
approaches_accuracy_regexes = {
    "tax": "ACC Sklearn",
    "evermann": "Accuracy"
}

# These regexes allow us to delete parts of the filename that are not relevant
approaches_clean_log_regexes = {
    "tax": "_next_event.log",
    "evermann": ".xes.txt"
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

available_logs = set()
for directory in directories:
    approach = directory.split("/")[0]
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
                    if re.match(accuracy_regex, line) is not None:
                        accuracy_line = line
                        break
                accuracy = float(accuracy_line.split(":")[-1])
                clean_filename = file.replace(approaches_clean_log_regexes[approach], "")
                parse_groups = re.match(log_regex, clean_filename)
                fold, variation, log = parse_groups.groups()
                available_logs.add(log)
                store_results(results, approach, log, fold, variation, accuracy)

print("Available logs: ", available_logs)
t_results = {}
# Perform paired t-test and retrieve results
for approach_A, approach_B in itertools.combinations(results.keys(), 2):
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

            s2 = (p1 - p_mean) ** 2 + (p2 - p_mean) ** 2 # Variance of the ith replication
            s2_list.append(s2)

        t_statistic = first_p / math.sqrt((1 / len(s2_list)) * sum(s2_list))
        # For some reason you must multiply by 2
        # I think it is because you are performing a two-sided test
        # https://github.com/rasbt/mlxtend/blob/7c329e40fab7659c3dd9b77f31a55db6f2400521/mlxtend/evaluate/ttest.py#L335
        p_value = t.sf(t_statistic, len(s2_list)) * 2
        print("T statistic: ", t_statistic)
        print("P value: ", p_value)
        t_results[((approach_A, approach_B), log)] = (t_statistic, p_value)

    print("T results: ", t_results)
