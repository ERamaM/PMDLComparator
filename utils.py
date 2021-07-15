from dateutil.tz import tzutc
import shutil

from sklearn.model_selection import KFold
from numpy import timedelta64, float64
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.log.importer.csv import factory as csv_import_factory
from pm4py.objects.log.exporter.csv import factory as csv_exporter
from pm4py.objects.log.exporter.xes import factory as xes_exporter
from pm4py.objects.conversion.log import factory as conversion_factory
import os
import pandas as pd
from pathlib import Path
import glob
import datetime
import shutil
import yaml, json
import random


def convert_xes_to_csv(file, output_folder, perform_lifecycle_trick=True, fill_na=None):
    """
    Load a xes file and convert it to csv.
    :param file: Full path of the xes file to convert
    :param output_folder: Output folder where to store the converted csv.
    :return: Both the csv file name and the full path of the converted csv
    """
    xes_path = file
    csv_file = Path(file).stem.split(".")[0] + ".csv"
    csv_path = os.path.join(output_folder, csv_file)
    log = xes_import_factory.apply(xes_path, parameters={"timestamp_sort": True})
    csv_exporter.export(log, csv_path)

    pd_log = pd.read_csv(csv_path)

    if fill_na is not None:
        pd_log.fillna(fill_na, inplace=True)
        pd_log.replace("-", fill_na, inplace=True)
        import numpy as np
        pd_log.replace(np.nan, fill_na)

    # Use integers always for case identifiers.
    # We need this to make a split that is equal for every dataset
    pd_log["case:concept:name"] = pd.Categorical(pd_log["case:concept:name"])
    pd_log["case:concept:name"] = pd_log["case:concept:name"].cat.codes

    unique_lifecycle = pd_log[XES_Fields.LIFECYCLE_COLUMN].unique()
    print("Unique lifecycle: ", unique_lifecycle, ". For log: ", file)
    if len(unique_lifecycle) > 1 and perform_lifecycle_trick:
        pd_log[XES_Fields.ACTIVITY_COLUMN] = pd_log[XES_Fields.ACTIVITY_COLUMN].astype(str) + "+" + pd_log[XES_Fields.LIFECYCLE_COLUMN]
        print("HEAD: ", pd_log.head(10))
    pd_log.to_csv(csv_path, encoding="utf-8")


    return csv_file, csv_path


def augment_xes_end_activity_to_csv(file, output_folder):
    """
    Augment a process log stored in xes format with and end of case token ([EOC])
    :param file: Input xes process log
    :param output_folder: Output folder where to store the augmented log
    :return: Both the csv file name and the full path of the converted csv
    """
    csv_file, csv_path = convert_xes_to_csv(file, output_folder)

    dataframe = pd.read_csv(csv_path)

    #group = dataframe.groupby(XES_Fields.CASE_COLUMN, as_index=False, sort=False)
    # Get the last rows (last event of each case, since its ordered by timestamp)
    #last_rows = group.last()
    #last_rows[XES_Fields.ACTIVITY_COLUMN] = "[EOC]"
    #final_dataframe = pd.concat([dataframe, last_rows])

    groups = [pandas_df for _, pandas_df in dataframe.groupby(XES_Fields.CASE_COLUMN, sort=False)]

    for i, group in enumerate(groups):
        last_rows = group[-1:].copy()
        last_rows[XES_Fields.ACTIVITY_COLUMN] = "[EOC]"
        groups[i] = pd.concat([group, last_rows])

    final_dataframe = pd.concat(groups, sort=False).reset_index(drop=True)
    
    final_dataframe.to_csv(csv_path, sep=",", index=False)

    return csv_file, csv_path


def convert_csv_to_xes(file, output_folder, extension):
    """
    Convert a csv log into a xes formated log.
    :param file: Full path of the csv file
    :param output_folder: Output folder where to store the converted process log.
    :param extension: Extension used to save the file. If the extension is "xes.gz" (EXTENSIONS.XES_COMPRESSED) the log will be compressed into a gz file. If the extension is other, it won't be compressed.
    :return: Both the converted xes file name and the xes path.
    """
    csv_path = file
    xes_file = Path(file).stem.split(".")[0] + EXTENSIONS.XES
    xes_path = os.path.join(output_folder, xes_file)
    log = csv_import_factory.apply(csv_path,
                                   parameters={"timestamp_sort": True, "timest_columns": XES_Fields.TIMESTAMP_COLUMN})
    if extension == EXTENSIONS.XES_COMPRESSED:
        xes_exporter.export_log(log, xes_path, parameters={"compress": True})
    else:
        xes_exporter.export_log(log, xes_path)

    if extension == EXTENSIONS.XES_COMPRESSED:
        xes_file += ".gz"
        xes_path += ".gz"
    return xes_file, xes_path


def convert_csv_to_json(file, output_folder, attributes, timestamp_format, prettify=False):
    """
    Convert a csv process log into a json file readable by the Hinkka et al. appoach.
    :param file: Full path to the csv log file
    :param output_folder: Output folder where to store the converted json
    :param attributes: List of attributes to add to the converted log.
    :param timestamp_format: Timestamp format of the input csv log (one of the "Timestamp_Format")
    :param prettify: Whether to prettify the log. A value of True padds the output json so as to it is readable.
    :return: Full path of the json file
    """
    csv_path = file
    json_log = {}
    log_df = pd.read_csv(csv_path)
    unique_activities = log_df[XES_Fields.ACTIVITY_COLUMN].unique().tolist()
    # List of activities and its ids
    activity_list = [{"name": activity, "id": i} for i, activity in enumerate(unique_activities)]
    activity_map = {}
    for i, activity in enumerate(unique_activities):
        activity_map[activity] = i
    json_log["activities"] = activity_list
    # List of attributes
    json_log["attributes"] = {}
    json_log["attributes"]["event"] = attributes
    # Note: no case attributes is taken into account
    json_log["attributes"]["case"] = []

    # Cases
    json_log["cases"] = []
    groups = [pandas_df for _, pandas_df in log_df.groupby(XES_Fields.CASE_COLUMN, sort=False)]
    fields = list(log_df)
    fields.remove(XES_Fields.CASE_COLUMN)
    for group in groups:
        case = {}
        case["t"] = []
        case["a"] = [] # Empty case attributes
        case["id"] = str(group.iloc[0][XES_Fields.CASE_COLUMN])
        # The kfold function returns the indices of the array but not the array contents itself.
        # Since the case ids are always positive sequential integers this does not matter except in the
        # SEPSIS log where there is a id = -1. For now, skip it.
        if case["id"] == "-1":
            continue
        """
        print("=========")
        print("Case id: ", case["id"])
        print(group)
        print("==========")
        """
        for idx, event in group.iterrows():
            case["n"] = str(event[XES_Fields.CASE_COLUMN]) # For some reason is a str
            j_event = []
            for field in fields:
                if field == XES_Fields.TIMESTAMP_COLUMN:
                    j_event.append("/Date(" + str(int(datetime.datetime.timestamp(
                        datetime.datetime.strptime(event[field], timestamp_format)))) + ")/")
                elif field == XES_Fields.ACTIVITY_COLUMN:
                    j_event.append(activity_map[event[field]])
                else:
                    j_event.append(event[field])
            case["t"].append(j_event)
        json_log["cases"].append(case)

    indent = 4 if prettify else None
    obj = json.dumps(json_log, indent=indent)
    json_path = os.path.join(output_folder, Path(file).stem + EXTENSIONS.JSON)
    with open(json_path, "w") as f:
        f.write(obj)
    return json_path


def create_tmp():
    """
    Create a temporal directory to store the temporal converted logs
    :return:
    """
    if not os.path.exists("tmp"):
        os.mkdir("tmp")


def delete_tmp():
    """
    Delete the temporal directory and all its contents
    :return:
    """
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")


class Timestamp_Formats:
    """
    Timestamp formats supported by the approaches. This timestamp formats are used whenever a timestamp format is required.
    """
    TIMESTAMP_FORMAT_YMDHMS_DASH = "%Y-%m-%d %H:%M:%S"
    TIMESTAMP_FORMAT_DAYS = "d"  # Used by: pasquadibisceglie
    TIMESTAMP_FORMAT_YMDHMS_SLASH = "%Y/%m/%d %H:%M:%S.%f"  # Used by: mauro"
    TIMESTAMP_FORMAT_YMDHMSf_DASH_T = "%Y-%m-%dT%H:%M:%S.%f"


class XES_Fields:
    """
    Supported xes fields that may be present in a xes log.
    """
    CASE_COLUMN = "case:concept:name"
    ACTIVITY_COLUMN = "concept:name"
    TIMESTAMP_COLUMN = "time:timestamp"
    LIFECYCLE_COLUMN = "lifecycle:transition"
    RESOURCE_COLUMN = "org:resource"


class EXTENSIONS:
    """
    Process log extensions
    """
    CSV = ".csv"
    XES = ".xes"
    XES_COMPRESSED = ".xes.gz"
    JSON = ".json"


def select_columns(file, input_columns, category_columns, timestamp_format, output_columns, categorize=False,
                   fill_na=None, francescomarino_fix=None, save_category_assignment=None):
    """
    Select columns from CSV converted from XES
    :param file: csv file
    :param input_columns: array with the columns to be selected. If none selects every column
    :param timestamp_format: timestamp format to be converted
    :param output_columns: dictionary with the assignment of renaming the columns. If none, no renaming is done
    :param francescomarino_fix: perform a fix over the case identifiers to allow the usage of the partitions in RuM (Declare Miner in ProM)
    :return: overwrites the csv file with the subselected csv
    """
    dataset = pd.read_csv(file)

    if fill_na is not None:
        dataset = dataset.fillna(fill_na)

    if input_columns is not None:
        dataset = dataset[input_columns]

    timestamp_column = XES_Fields.TIMESTAMP_COLUMN
    dataset[timestamp_column] = pd.to_datetime(dataset[timestamp_column], utc=True)
    if timestamp_format == Timestamp_Formats.TIMESTAMP_FORMAT_DAYS:
        # If the timestamp format is in days (pasquadibisceglie), get the timestamp, remove the localization,
        # get the number of seconds since 1/1/1970 and get the number of days (with decimals) from there
        dataset[timestamp_column] = (dataset[timestamp_column].dt.tz_localize(None) - datetime.datetime(1970, 1,
                                                                                                        1)).dt.total_seconds() / 86400
    else:
        dataset[timestamp_column] = dataset[timestamp_column].dt.strftime(timestamp_format)

    if categorize:
        for category_column in category_columns:
            if category_column == XES_Fields.ACTIVITY_COLUMN:
                category_list = dataset[category_column].astype("category").cat.categories.tolist()
                category_dict = {c : i for i, c in enumerate(category_list)}
                if save_category_assignment is None:
                    print("Activity assignment: ", category_dict)
                else:
                    file_name = Path(file).name
                    with open(os.path.join(save_category_assignment, file_name), "w") as fw:
                        fw.write(str(category_dict))
            dataset[category_column] = dataset[category_column].astype("category").cat.codes

    # For reasons unknown, using this scheme of partitioning is not good for RuM and, in general, for loading xes Java based systems.
    # The cause of failure is that the case identifiers are integers and it fails in doing some casting.
    # To fix that append a fixed string to the case identifiers and convert the field to a string.
    if francescomarino_fix:
        dataset[XES_Fields.CASE_COLUMN] = "FIX" + dataset[XES_Fields.CASE_COLUMN].astype(str)
        # Furthermore, the LTL checker from rum does not work correctly with lifecycle transitions.
        # in fact, it may happen that the LTL checker detects every trace as non-conformant if the
        # pair of activities involucrated have different lifecycle:transitions
        # If we switch every lifecycle:transition to complete and make artificial activities as
        # each pair of concept:name and lifecycle:transition we can circunvent this.
        dataset[XES_Fields.LIFECYCLE_COLUMN] = "Complete"

    if output_columns is not None:
        dataset.rename(
            output_columns,
            axis="columns",
            inplace=True
        )

    dataset.to_csv(file, sep=",", index=False)


def reorder_columns(file, ordered_columns):
    """
    Reorder the columns of the dataset.
    :param file: Data file
    :param ordered_columns: Array with the ordered columns of the df.
    If the column array size is less than the total number of columns of the dataset,
     only that columns are ordered and the rest is left alone.
    :return:
    """
    df = pd.read_csv(file)
    df = df.reindex(columns=(ordered_columns + list([a for a in df.columns if a not in ordered_columns])))
    df.to_csv(file, sep=",", index=False)


def load_attributes_from_file(filename, log_name):
    """
    Loads the dictionary of attributes from the logs from the file "attributes.yaml"
    :param filename: Attribute filename
    :param log_name: Log of which to load the attributes
    :return: Array of attributes
    """
    with open(filename) as yaml_file:
        data = yaml.safe_load(yaml_file)
        if log_name.find(".") != -1:
            name = log_name.split(".")[0]
        else:
            name = log_name

        attributes = data[name]
    return attributes


def split_train_val_test(file, output_directory, case_column, do_train_val=False):
    """
    Perform the splits for a 5x2cv with validation set
    :param file:
    :param output_directory:
    :param do_train_val:
    :return:
    """
    pandas_init = pd.read_csv(file)
    pd.set_option('display.expand_frame_repr', False)
    # print(str(pandas_init.head(50)))

    # Disable the sorting. Otherwise it would mess with the order of the timestamps
    unique_case_ids = list(pandas_init[case_column].unique())

    train_paths = []
    val_paths = []
    test_paths = []
    train_val_paths = []

    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    indexes = sorted(unique_case_ids)
    #print("Unique case ids: ", sorted(unique_case_ids))
    variation = 0
    splits = kfold.split(indexes)
    repetition = 0
    for train_index, test_index in splits:
        # Select which traces are we going to use for this split
        #random.Random(seeds[repetition]).shuffle(indexes)
        # If it is the second time on the same repetition, reverse the index array to get test-train, instead of train-test
        if variation == 1:
            indexes = list(reversed(indexes))

        # From the train size, select the 20% of the train size (10% of the total) as the validation set
        train_size = round(len(train_index) * 0.8)

        train_groups = train_index[:train_size]
        val_groups = train_index[train_size:]
        test_groups = test_index

        """
        print("===============")
        print("Full train: ", train_index, " Full test: ", test_index)
        print("Train: ", train_groups)
        print("Val: ", val_groups)
        print("Test: ", test_groups)
        print("===============")
        """

        train_list = [pandas_init[pandas_init[case_column] == train_g] for train_g in train_groups]
        val_list = [pandas_init[pandas_init[case_column] == val_g] for val_g in val_groups]
        test_list = [pandas_init[pandas_init[case_column] == test_g] for test_g in test_groups]

        # Disable the sorting. Otherwise it would mess with the order of the timestamps
        train = pd.concat(train_list, sort=False).reset_index(drop=True)
        val = pd.concat(val_list, sort=False).reset_index(drop=True)
        test = pd.concat(test_list, sort=False).reset_index(drop=True)

        train_path = os.path.join(output_directory, "train_fold" + str(repetition) + "_variation" + str(variation) + "_" + Path(file).stem + ".csv")
        val_path = os.path.join(output_directory, "val_fold" + str(repetition) + "_variation" + str(variation) + "_" + Path(file).stem + ".csv")
        test_path = os.path.join(output_directory, "test_fold" + str(repetition) + "_variation" + str(variation) + "_" + Path(file).stem + ".csv")

        train_val_path = None
        if do_train_val:
            train_val_groups = train_index
            train_val_list = [pandas_init[pandas_init[case_column] == train_val_g] for train_val_g in train_val_groups]
            train_val = pd.concat(train_val_list, sort=False).reset_index(drop=True)
            train_val_path = os.path.join(output_directory, "train_val_fold" + str(repetition) + "_variation" + str(variation) + "_" + Path(file).stem + ".csv")
            train_val.to_csv(train_val_path, index=False)
            train_val_paths.append(train_val_path)

        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)
        train_paths.append(train_path)
        val_paths.append(val_path)
        test_paths.append(test_path)

        repetition += 1

    if do_train_val:
        return file, train_paths, val_paths, test_paths, train_val_paths
    else:
        return file, train_paths, val_paths, test_paths


def move_files(files_to_move, output_directory):
    """
    Move the set of files to the output directory.
    This function also moves the train/val/test files automatically.
    :param file: Path of the base file to move
    :param output_directory: directory to move the files
    :param extension: extension of the files to move
    :return:
    """

    # name = Path(file).stem + extension
    def _move_if_exists(input, output_dir):
        head, tail = os.path.split(input)
        if os.path.exists(input):
            os.replace(input, os.path.join(output_directory, tail))

    for file in files_to_move:
        _move_if_exists(file, output_directory)

def make_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def gather_statistics(log, abbreviate_times=True):
    """
    Gather useful statistics from a process log.
    :param log: Path to Xes formatted process log.
    :param abbreviate_times: True: convert the time related measures to days and round the values to two decimals.
    :return: Dataframe of gathered statistics.
    """
    pd.set_option('display.expand_frame_repr', False)
    csv_file, csv_path = convert_xes_to_csv(log, "./tmp")
    log_df = pd.read_csv(csv_path)
    log_df[XES_Fields.TIMESTAMP_COLUMN] = pd.to_datetime(log_df[XES_Fields.TIMESTAMP_COLUMN], utc=True)
    log_name = Path(log).stem

    group_by_case = log_df.groupby(XES_Fields.CASE_COLUMN)

    # It does not match with https://arxiv.org/pdf/1805.02896.pdf
    # It matches https://kodu.ut.ee/~dumas/pubs/bpm2019lstm.pdf
    n_cases = len(group_by_case)
    n_activities = len(log_df.groupby(XES_Fields.ACTIVITY_COLUMN))
    n_events = len(log_df)
    avg_case_length = group_by_case[XES_Fields.ACTIVITY_COLUMN].count().mean()
    max_case_length = group_by_case[XES_Fields.ACTIVITY_COLUMN].count().max()
    avg_event_duration = group_by_case[XES_Fields.TIMESTAMP_COLUMN].diff().mean()
    max_event_duration = group_by_case[XES_Fields.TIMESTAMP_COLUMN].diff().max()
    first_and_last_timestamps_per_case = group_by_case[XES_Fields.TIMESTAMP_COLUMN].agg(
        ["first", "last"])
    avg_case_duration = (
            first_and_last_timestamps_per_case["last"] - first_and_last_timestamps_per_case["first"]).mean()
    max_case_duration = (first_and_last_timestamps_per_case["last"] - first_and_last_timestamps_per_case["first"]).max()
    variants = group_by_case[XES_Fields.ACTIVITY_COLUMN].agg("->".join).nunique()
    print("N_cases: ", n_cases)
    print("N_activities: ", n_activities)
    print("N_events: ", n_events)
    print("Avg case length: ", avg_case_length)
    print("Max case length: ", max_case_length)
    print("Avg event duration: ", avg_event_duration)
    print("Max event duration: ", max_event_duration)
    print("Avg case duration: ", avg_case_duration)
    print("Max case duration: ", max_case_duration)
    print("Variants: ", variants)
    final_df = pd.DataFrame(
        [[log_name, n_cases, n_activities, n_events, avg_case_length, max_case_length, avg_event_duration,
          max_event_duration, avg_case_duration, max_case_duration, variants]],
        columns=["Log name", "Number of cases", "Number of activities", "Number of events", "Average case length",
                 "Maximum case length", "Average event duration", "Maximum event duration", "Average case duration",
                 "Maximum case duration", "Distinct variants"]
    )
    if abbreviate_times:
        for column in final_df.columns:
            if final_df[column].dtype == "timedelta64[ns]":
                final_df[column] = round(final_df[column].dt.total_seconds() / 86400, 2)
            if final_df[column].dtype == float64:
                final_df[column] = round(final_df[column], 2)

    return final_df
