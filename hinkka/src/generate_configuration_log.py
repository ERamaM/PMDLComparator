
base_json = {
    "num_callbacks": 100,
    "num_epochs_per_iteration": 0.1,
    "test_name": "",

    "case_clustering_include_activity_occurrences": False,
    "case_clustering_include_case_attributes": True,
    "use_single_event_clustering": False,
    "include_activity_occurrences_as_raw_case_attributes": False,
    "use_single_value_for_duration": True,
    "split_traces_to_prefixes": True,
    "predict_next_activity": True,
    "cross-validation-splits": None,
    "create-others-token": False,
    "max_num_cases_in_training": 10000000000,
    "max_num_traces_in_training": 7500000000,
    "max_num_traces_in_training_test": 2500000000,
    "max_num_traces_in_testing": 10000000000,

    "case_clustering_method": "xmeans",
    "event_clustering_method": "xmeans",

    "disable_activity_labels": False,
    "disable_durations": True,
    "disable_case_attributes": True,
    "disable_raw_case_attributes": True,

    "ignore_values_threshold_for_case_attributes": 0,
    "ignore_values_threshold_for_event_attributes": 0,

    "for_each": [],

    "runs": [
        {
            "for_each": [
                {
                    "max_num_event_clusters": 20
                }
            ],
            "runs": [
                {
                    "case_name": "both-event-attributes",
                    "disable_raw_event_attributes": False,
                    "disable_event_attributes": False
                }
            ]
        }
    ]
}

import os
import json
import argparse

parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--dataset", help="Raw dataset to generate configuration for")
arguments = parser.parse_args()

filter_dataset = ""
if arguments.dataset is not None:
    filter_dataset = arguments.dataset

for log in os.listdir("./testdata"):
    if "train" in log and (filter_dataset == "" or filter_dataset in log):
        logname = log.replace("train_", "").replace(".csv", "")
        print("Logname: ", logname)
        """
        if filter_dataset == "bpi_challenge_2013_incidents":
            with open("./config/" + logname, "w") as f:
                json.dump(base_json, f, indent=4)
        """
        base_json["for_each"].append(
            {
                "input_filename" : logname,
                "dataset_name" : logname
            }
        )

        base_json["for_each"].append(
            {
                "model_filename" : "testdata/modelstest_" + logname + "-.model",
                "test_filename" : logname
            }
        )
        """
        if filter_dataset == "bpi_challenge_2013_incidents":
            with open("./config/" + logname + ".json", "w") as f:
                json.dump(base_json, f, indent=4)
        """

    if filter_dataset != "":
        config_file_name = "custom_" + filter_dataset + ".json"
    else:
        config_file_name = "custom_logs.json"
    """
    if filter_dataset != "bpi_challenge_2013_incidents":
    """
    with open("./config/" + config_file_name, "w") as f:
        json.dump(base_json, f, indent=4)

