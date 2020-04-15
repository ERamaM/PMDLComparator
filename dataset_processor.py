import argparse
from utils import *

parser = argparse.ArgumentParser(description="Prepare datasets for multiple SOTA DL-PPM studies")
parser.add_argument("--net", help="Neural net to prepare the data for", required=True)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--dataset", help="Raw dataset to prepare")
group.add_argument("--batch", help="Batch process the selected folder")
arguments = parser.parse_args()

dataset_list = []
if arguments.batch:
    files = os.listdir(arguments.batch)
    for dataset in files:
        dataset_list.append(os.path.join("./", arguments.batch, dataset))

else:
    dataset_list = [arguments.dataset]

print("Dataset list: ", dataset_list)

create_tmp()

if arguments.net == "pasquadibisceglie":
    for xes in dataset_list:
        csv_file, csv_path = augment_xes_end_activity_to_csv(xes, "./tmp")
        output_columns = {
            XES_Fields.CASE_COLUMN : "CaseID",
            XES_Fields.ACTIVITY_COLUMN : "Activity",
            XES_Fields.TIMESTAMP_COLUMN : "Timestamp"
        }
        select_columns(
            csv_path,
            input_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN, XES_Fields.TIMESTAMP_COLUMN],
            category_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN],
            timestamp_format=Timestamp_Formats.TIMESTAMP_FORMAT_DAYS,
            output_columns=output_columns
        )
        csv_path, train_path, val_path, test_path = split_train_val_test(csv_path, "./tmp", "CaseID")
        make_dir_if_not_exists("ImagePPMiner/dataset")
        # Move from tmp to the corresponding folder
        move_files(csv_path, "ImagePPMiner/dataset", EXTENSIONS.CSV)

elif arguments.net == "mauro":
    for xes in dataset_list:
        #csv_file, csv_path = augment_xes_end_activity_to_csv(xes, "./tmp")
        csv_file, csv_path = augment_xes_end_activity_to_csv(xes, "./tmp")

        output_columns = {
            XES_Fields.CASE_COLUMN : "CaseID",
            XES_Fields.ACTIVITY_COLUMN : "Activity",
            XES_Fields.TIMESTAMP_COLUMN : "Timestamp"
        }
        select_columns(
            csv_path,
            input_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN, XES_Fields.TIMESTAMP_COLUMN],
            category_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN],
            timestamp_format=Timestamp_Formats.TIMESTAMP_FORMAT_YMDHMS_SLASH,
            output_columns=output_columns
        )
        csv_path, train_path, val_path, test_path = split_train_val_test(csv_path, "./tmp", "CaseID")
        make_dir_if_not_exists("nnpm/data")
        make_dir_if_not_exists("nnpm/results")
        move_files(csv_path, "nnpm/data", EXTENSIONS.CSV)

elif arguments.net == "tax":
    for xes in dataset_list:
        # Tax already performs the augmentation in their script
        # so there is no need to perform it here
        csv_file, csv_path = convert_xes_to_csv(xes, "./tmp")

        output_columns = {
            XES_Fields.CASE_COLUMN : "CaseID",
            XES_Fields.ACTIVITY_COLUMN : "ActivityID",
            XES_Fields.TIMESTAMP_COLUMN : "CompleteTimestamp"
        }
        select_columns(
            csv_path,
            input_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN, XES_Fields.TIMESTAMP_COLUMN],
            category_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN],
            timestamp_format=Timestamp_Formats.TIMESTAMP_FORMAT_YMDHMS_DASH,
            output_columns=output_columns
        )
        csv_path, train_path, val_path, test_path = split_train_val_test(csv_path, "./tmp", "CaseID")
        make_dir_if_not_exists("tax/data")
        make_dir_if_not_exists("tax/code/results")
        make_dir_if_not_exists("tax/code/models")
        move_files(csv_path, "tax/data", EXTENSIONS.CSV)
else:
    print("Unrecognized approach")

delete_tmp()



