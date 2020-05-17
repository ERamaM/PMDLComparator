import argparse
from pathlib import Path
from utils import *

parser = argparse.ArgumentParser(description="Prepare datasets for multiple SOTA DL-PPM studies")
action = parser.add_mutually_exclusive_group(required=True)
action.add_argument("--statistics", help="Gather statistics from the event logs", action="store_true")
action.add_argument("--net", help="Neural net to prepare the data for")
processing_mode = parser.add_mutually_exclusive_group(required=True)
processing_mode.add_argument("--dataset", help="Raw dataset to prepare")
processing_mode.add_argument("--batch", help="Batch process the selected folder")
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
if arguments.net:
    if arguments.net == "pasquadibisceglie":
        for xes in dataset_list:
            csv_file, csv_path = augment_xes_end_activity_to_csv(xes, "./tmp")
            output_columns = {
                XES_Fields.CASE_COLUMN: "CaseID",
                XES_Fields.ACTIVITY_COLUMN: "Activity",
                XES_Fields.TIMESTAMP_COLUMN: "Timestamp"
            }
            select_columns(
                csv_path,
                input_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN, XES_Fields.TIMESTAMP_COLUMN],
                category_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN],
                timestamp_format=Timestamp_Formats.TIMESTAMP_FORMAT_DAYS,
                output_columns=output_columns, categorize=True
            )
            csv_path, train_path, val_path, test_path = split_train_val_test(csv_path, "./tmp", "CaseID")
            make_dir_if_not_exists("ImagePPMiner/dataset")
            # Move from tmp to the corresponding folder
            move_files(csv_path, "ImagePPMiner/dataset", EXTENSIONS.CSV)

    elif arguments.net == "mauro":
        for xes in dataset_list:
            # csv_file, csv_path = augment_xes_end_activity_to_csv(xes, "./tmp")
            csv_file, csv_path = augment_xes_end_activity_to_csv(xes, "./tmp")

            output_columns = {
                XES_Fields.CASE_COLUMN: "CaseID",
                XES_Fields.ACTIVITY_COLUMN: "Activity",
                XES_Fields.TIMESTAMP_COLUMN: "Timestamp"
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
                XES_Fields.CASE_COLUMN: "CaseID",
                XES_Fields.ACTIVITY_COLUMN: "ActivityID",
                XES_Fields.TIMESTAMP_COLUMN: "CompleteTimestamp"
            }
            select_columns(
                csv_path,
                input_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN, XES_Fields.TIMESTAMP_COLUMN],
                category_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN],
                timestamp_format=Timestamp_Formats.TIMESTAMP_FORMAT_YMDHMS_DASH,
                output_columns=output_columns, categorize=True
            )
            csv_path, train_path, val_path, test_path = split_train_val_test(csv_path, "./tmp", "CaseID")
            make_dir_if_not_exists("tax/data")
            make_dir_if_not_exists("tax/code/results")
            make_dir_if_not_exists("tax/code/models")
            move_files(csv_path, "tax/data", EXTENSIONS.CSV)
    elif arguments.net == "evermann":
        for xes in dataset_list:
            print("Process: ", xes)
            make_dir_if_not_exists("evermann/data")
            make_dir_if_not_exists("evermann/models")
            make_dir_if_not_exists("evermann/results")
            csv_file, csv_path = convert_xes_to_csv(xes, "./tmp")
            csv_path, train_path, val_path, test_path = split_train_val_test(csv_path, "./tmp", XES_Fields.CASE_COLUMN)
            xes_file, xes_path = convert_csv_to_xes(csv_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            train_file, train_path = convert_csv_to_xes(train_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            val_file, val_path = convert_csv_to_xes(val_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            test_file, test_path = convert_csv_to_xes(test_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            move_files(xes_path, "evermann/data", EXTENSIONS.XES_COMPRESSED)
    elif arguments.net == "thai":
            for xes in dataset_list:
                print("Process: ", xes)
                make_dir_if_not_exists("MAED/data")
                make_dir_if_not_exists("MAED/models")
                make_dir_if_not_exists("MAED/results")
                csv_file, csv_path = convert_xes_to_csv(xes, "./tmp")
                csv_path, train_path, val_path, test_path = split_train_val_test(csv_path, "./tmp",
                                                                                 XES_Fields.CASE_COLUMN)
                xes_file, xes_path = convert_csv_to_xes(csv_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
                train_file, train_path = convert_csv_to_xes(train_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
                val_file, val_path = convert_csv_to_xes(val_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
                test_file, test_path = convert_csv_to_xes(test_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
                move_files(xes_path, "MAED/data", EXTENSIONS.XES_COMPRESSED)
    elif arguments.net == "theis":
        for xes in dataset_list:
            print("Process: ", xes)
            make_dir_if_not_exists("PyDREAM-NAP/best_models")
            make_dir_if_not_exists("PyDREAM-NAP/enhanced_pns")
            make_dir_if_not_exists("PyDREAM-NAP/logs")
            make_dir_if_not_exists("PyDREAM-NAP/output_models")
            make_dir_if_not_exists("PyDREAM-NAP/model_checkpoints")
            make_dir_if_not_exists("PyDREAM-NAP/results")

            csv_file, csv_path = convert_xes_to_csv(xes, "./tmp")
            csv_path, train_path, val_path, test_path, train_val_path = split_train_val_test(csv_path, "./tmp",
                                                                             XES_Fields.CASE_COLUMN, do_train_val=True)
            xes_file, xes_path = convert_csv_to_xes(csv_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            train_file, train_path = convert_csv_to_xes(train_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            val_file, val_path = convert_csv_to_xes(val_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            test_file, test_path = convert_csv_to_xes(test_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            train_val_file, train_val_path = convert_csv_to_xes(train_val_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            move_files(xes_path, "PyDREAM-NAP/logs", EXTENSIONS.XES_COMPRESSED)
    elif arguments.net == "navarin":
        for xes in dataset_list:
            print("Process: ", xes)
            make_dir_if_not_exists("DALSTM/data")
            make_dir_if_not_exists("DALSTM/model")
            make_dir_if_not_exists("DALSTM/model/model_data/")
            make_dir_if_not_exists("DALSTM/results")
            csv_file, csv_path = convert_xes_to_csv(xes, "./tmp")
            # Load attribute list from file
            attributes = load_attributes_from_file("attributes.yaml", Path(xes).name)
            # Normalize columns
            select_columns(
                csv_path,
                input_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN, XES_Fields.TIMESTAMP_COLUMN] + attributes,
                category_columns=None,
                timestamp_format=Timestamp_Formats.TIMESTAMP_FORMAT_YMDHMS_DASH,
                output_columns=None, categorize=False
            )
            # Reorder columns
            reorder_columns(csv_path, [XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN, XES_Fields.TIMESTAMP_COLUMN])
            csv_path, train_path, val_path, test_path = split_train_val_test(csv_path, "./tmp", XES_Fields.CASE_COLUMN, do_train_val=False)
            move_files(csv_path, "DALSTM/data", EXTENSIONS.CSV)
    elif arguments.net == "camargo":
        for xes in dataset_list:
            """
            print("Process: ", xes)
            make_dir_if_not_exists("GenerativeLSTM/input_files")
            make_dir_if_not_exists("GenerativeLSTM/input_files/embedded_matix")
            csv_file, csv_path = convert_xes_to_csv(xes, "./tmp")
            csv_path, train_path, val_path, test_path = split_train_val_test(csv_path, "./tmp", XES_Fields.CASE_COLUMN)
            xes_file, xes_path = convert_csv_to_xes(csv_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            train_file, train_path = convert_csv_to_xes(train_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            val_file, val_path = convert_csv_to_xes(val_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            test_file, test_path = convert_csv_to_xes(test_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            move_files(xes_path, "GenerativeLSTM/input_files", EXTENSIONS.XES_COMPRESSED)
            """
            print("Process: ", xes)
            attributes = load_attributes_from_file("attributes.yaml", Path(xes).name)
            make_dir_if_not_exists("GenerativeLSTM/input_files")
            # Note the typo in "matix"
            make_dir_if_not_exists("GenerativeLSTM/input_files/embedded_matix")
            # Only process files with resources
            if XES_Fields.RESOURCE_COLUMN in attributes:
                csv_file, csv_path = convert_xes_to_csv(xes, "./tmp")
                # Load attribute list from file
                    # Normalize columns
                output_columns = {
                    XES_Fields.CASE_COLUMN: "caseid",
                    XES_Fields.ACTIVITY_COLUMN: "task",
                    XES_Fields.RESOURCE_COLUMN: "user",
                    XES_Fields.TIMESTAMP_COLUMN: "end_timestamp"
                }
                select_columns(
                    csv_path,
                    input_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN, XES_Fields.TIMESTAMP_COLUMN, XES_Fields.RESOURCE_COLUMN],
                    category_columns=None,
                    timestamp_format=Timestamp_Formats.TIMESTAMP_FORMAT_YMDHMS_DASH,
                    output_columns=output_columns, categorize=False
                )
                # Reorder columns
                reorder_columns(csv_path, ["caseid", "task", "user", "end_timestamp"])
                csv_path, train_path, val_path, test_path = split_train_val_test(csv_path, "./tmp", "caseid", do_train_val=False)
                move_files(csv_path, "GenerativeLSTM/input_files", EXTENSIONS.CSV)
    else:
        print("Unrecognized approach")


elif arguments.statistics:
    make_dir_if_not_exists("stats")
    df = None
    for xes in dataset_list:
        df_log = gather_statistics(xes)
        if df is None:
            df = df_log
        else:
            df = pd.concat([df, df_log])

    df.to_csv("stats/log_stats.csv", index=False)
else:
    print("Unrecognized command")
    print("Arguments: ", arguments)

delete_tmp()
