from dateutil.tz import tzutc
import shutil

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


def convert_xes_to_csv(file, output_folder):
    xes_path = file
    csv_file = Path(file).stem.split(".")[0] + ".csv"
    csv_path = os.path.join(output_folder, csv_file)
    log = xes_import_factory.apply(xes_path, parameters={"timestamp_sort": True})
    csv_exporter.export(log, csv_path)
    return csv_file, csv_path


def augment_xes_end_activity_to_csv(file, output_folder):
    csv_file, csv_path = convert_xes_to_csv(file, output_folder)

    dataframe = pd.read_csv(csv_path)
    group = dataframe.groupby(XES_Fields.CASE_COLUMN, as_index=False)
    # Get the last rows (last event of each case, since its ordered by timestamp)
    last_rows = group.last()
    last_rows[XES_Fields.ACTIVITY_COLUMN] = "[EOC]"

    final_dataframe = pd.concat([dataframe, last_rows]).sort_values(
        [XES_Fields.CASE_COLUMN, XES_Fields.TIMESTAMP_COLUMN])
    # final_dataframe.fillna(None)
    final_dataframe = final_dataframe.fillna("")
    print(final_dataframe.head(5))
    final_dataframe.to_csv(csv_path, sep=",", index=False)

    return csv_file, csv_path


def convert_csv_to_xes(file, output_folder, extension):
    csv_path = file
    xes_file = Path(file).stem.split(".")[0] + EXTENSIONS.XES
    xes_path = os.path.join(output_folder, xes_file)
    log = csv_import_factory.apply(csv_path, parameters={"timestamp_sort": True, "timest_columns" : XES_Fields.TIMESTAMP_COLUMN})
    if extension == EXTENSIONS.XES_COMPRESSED:
        xes_exporter.export_log(log, xes_path, parameters={"compress" : True})
    else:
        xes_exporter.export_log(log, xes_path)

    if extension == EXTENSIONS.XES_COMPRESSED:
        xes_file += ".gz"
        xes_path += ".gz"
    return xes_file, xes_path


def create_tmp():
    if not os.path.exists("tmp"):
        os.mkdir("tmp")


def delete_tmp():
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")


class Timestamp_Formats:
    TIMESTAMP_FORMAT_YMDHMS_DASH = "%Y-%m-%d %H:%M:%S"
    TIMESTAMP_FORMAT_DAYS = "d"  # Used by: pasquadibisceglie
    TIMESTAMP_FORMAT_YMDHMS_SLASH = "%Y/%m/%d %H:%M:%S.%f"  # Used by: mauro


class XES_Fields:
    CASE_COLUMN = "case:concept:name"
    ACTIVITY_COLUMN = "concept:name"
    TIMESTAMP_COLUMN = "time:timestamp"
    LIFECYCLE_COLUMN = "lifecycle:transition"


class EXTENSIONS:
    CSV = ".csv"
    XES = ".xes"
    XES_COMPRESSED = ".xes.gz"


def select_columns(file, input_columns, category_columns, timestamp_format, output_columns):
    """
    Select columns from CSV converted from XES
    :param file: csv file
    :param input_columns: array with the columns to be selected
    :param timestamp_format: timestamp format to be converted
    :param output_columns: dictionary with the assignment
    :return: overwrites the csv file with the subselected csv
    """
    dataset = pd.read_csv(file)

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

    for category_column in category_columns:
        dataset[category_column] = dataset[category_column].astype("category").cat.codes

    dataset.rename(
        output_columns,
        axis="columns",
        inplace=True
    )

    dataset.to_csv(file, sep=",", index=False)


def split_train_val_test(file, output_directory, case_column):
    """
    Split the TRACES of the log in a 64/16/20 fashion (first 80/20 and then again 80/20).
    We assume the input is a csv file.
    :param file:
    :param output_directory:
    :return:
    """
    pandas_init = pd.read_csv(file)
    pd.set_option('display.expand_frame_repr', False)
    # print(str(pandas_init.head(50)))

    # Disable the sorting. Otherwise it would mess with the order of the timestamps
    groups = [pandas_df for _, pandas_df in pandas_init.groupby(case_column, sort=False)]

    train_size = round(len(groups) * 0.64)
    val_size = round(len(groups) * 0.8)

    train_groups = groups[:train_size]
    val_groups = groups[train_size:val_size]
    test_groups = groups[val_size:]

    # Disable the sorting. Otherwise it would mess with the order of the timestamps
    train = pd.concat(train_groups, sort=False).reset_index(drop=True)
    val = pd.concat(val_groups, sort=False).reset_index(drop=True)
    test = pd.concat(test_groups, sort=False).reset_index(drop=True)

    train_path = os.path.join(output_directory, "train_" + Path(file).stem + ".csv")
    val_path = os.path.join(output_directory, "val_" + Path(file).stem + ".csv")
    test_path = os.path.join(output_directory, "test_" + Path(file).stem + ".csv")
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    return file, train_path, val_path, test_path

def copy_file(file, output_directory, extension):
    base_file = os.path.basename(file)
    dox_index = base_file.index(".")
    file_name = base_file[:dox_index]
    print("Copy file: ", file)
    shutil.copyfile(file, os.path.join(output_directory, file_name + extension))

def move_files(file, output_directory, extension):
    """
    Move the set of files to the output directory.
    This function also moves the train/val/test files automatically.
    :param file: Path of the base file to move
    :param output_directory:
    :return:
    """

    #name = Path(file).stem + extension
    basename = os.path.basename(file)
    dot_index = basename.index(".")
    name = basename[:dot_index] + extension
    input_directory = Path(file).parent
    input_train = os.path.join(input_directory, "train_" + name)
    input_val = os.path.join(input_directory, "val_" + name)
    input_test = os.path.join(input_directory, "test_" + name)

    def _move_if_not_exists(input, output_dir):
        stem = os.path.basename(input)
        if os.path.exists(input):
            os.replace(input, os.path.join(output_directory, stem))

    _move_if_not_exists(input_train, output_directory)
    _move_if_not_exists(input_val, output_directory)
    _move_if_not_exists(input_test, output_directory)
    _move_if_not_exists(file, output_directory)


def make_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def gather_statistics(log, abbreviate_times=True):
    pd.set_option('display.expand_frame_repr', False)
    csv_file, csv_path = convert_xes_to_csv(log, "./tmp")
    log_df = pd.read_csv(csv_path)
    log_df[XES_Fields.TIMESTAMP_COLUMN] = pd.to_datetime(log_df[XES_Fields.TIMESTAMP_COLUMN], utc=True)
    log_name = Path(log).stem

    group_by_case = log_df.groupby(XES_Fields.CASE_COLUMN)

    # WTF: it does not match with https://arxiv.org/pdf/1805.02896.pdf
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
