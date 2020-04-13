from dateutil.tz import tzutc
import shutil
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.log.importer.csv import factory as csv_import_factory
from pm4py.objects.log.exporter.csv import factory as csv_exporter
from pm4py.objects.log.exporter.xes import factory as xes_exporter
import os
import pandas as pd
from pathlib import Path
import glob
import datetime


def convert_xes_to_csv(file, output_folder):
    print("Processing: ", file)
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

    final_dataframe = pd.concat([dataframe, last_rows]).sort_values([XES_Fields.CASE_COLUMN, XES_Fields.TIMESTAMP_COLUMN])
    #final_dataframe.fillna(None)
    final_dataframe = final_dataframe.fillna("")
    print(final_dataframe.head(5))
    final_dataframe.to_csv(csv_path, sep=",", index=False)

    return csv_file, csv_path

def convert_csv_to_xes(file, output_folder):
    print("Processing: ", file)
    csv_path = file
    xes_file = Path(file).stem.split(".")[0] + ".xes"
    xes_path = os.path.join(output_folder, xes_file)
    log = csv_import_factory.apply(csv_path, parameters={"timestamp_sort": True})
    xes_exporter.export_log(log, xes_path)
    return xes_file, xes_path


def create_tmp():
    if not os.path.exists("tmp"):
        os.mkdir("tmp")


def delete_tmp():
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")


class Timestamp_Formats:
    TIMESTAMP_FORMAT_YMDHMS = "%Y-%m-%d %H:%M:%S"
    TIMESTAMP_FORMAT_DAYS = "d"

class XES_Fields:
    CASE_COLUMN = "case:concept:name"
    ACTIVITY_COLUMN = "concept:name"
    TIMESTAMP_COLUMN = "time:timestamp"
    LIFECYCLE_COLUMN = "lifecycle:transition"


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
    if not timestamp_column == Timestamp_Formats.TIMESTAMP_FORMAT_DAYS:
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
