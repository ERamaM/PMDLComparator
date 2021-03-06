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
            #csv_file, csv_path = convert_xes_to_csv(xes, "./tmp")

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
                make_dir_if_not_exists("MAED-TaxIntegration/busi_task")
                make_dir_if_not_exists("MAED-TaxIntegration/busi_task/data")

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
                move_files(csv_path, "MAED-TaxIntegration/busi_task/data", EXTENSIONS.CSV)
    elif arguments.net == "theis":
        for xes in dataset_list:
            print("Process: ", xes)
            make_dir_if_not_exists("PyDREAM-NAP/best_models")
            make_dir_if_not_exists("PyDREAM-NAP/enhanced_pns")
            make_dir_if_not_exists("PyDREAM-NAP/logs")
            make_dir_if_not_exists("PyDREAM-NAP/output_models")
            make_dir_if_not_exists("PyDREAM-NAP/model_checkpoints")
            make_dir_if_not_exists("PyDREAM-NAP/results")

            csv_file, csv_path = convert_xes_to_csv(xes, "./tmp", perform_lifecycle_trick=False, fill_na="UNK")
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
                # The code fails with unknown resources
                select_columns(
                    csv_path,
                    input_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN, XES_Fields.TIMESTAMP_COLUMN, XES_Fields.RESOURCE_COLUMN],
                    category_columns=None,
                    timestamp_format=Timestamp_Formats.TIMESTAMP_FORMAT_YMDHMSf_DASH_T,
                    output_columns=output_columns, categorize=False, fill_na="UNK"
                )
                # Reorder columns
                reorder_columns(csv_path, ["caseid", "task", "user", "end_timestamp"])
                # Do not split in train-val-test. The code already does that in the same manner as here.
                move_files(csv_path, "GenerativeLSTM/input_files", EXTENSIONS.CSV)
            else:
                print(xes + " does not have resources.")

    elif arguments.net == "francescomarino":
        for xes in dataset_list:
            print("Process: ", xes)
            make_dir_if_not_exists("Process-Sequence-Prediction-with-A-priori-knowledge/models")
            make_dir_if_not_exists("Process-Sequence-Prediction-with-A-priori-knowledge/results")
            make_dir_if_not_exists("Process-Sequence-Prediction-with-A-priori-knowledge/data")
            make_dir_if_not_exists("Process-Sequence-Prediction-with-A-priori-knowledge/data/declare_miner_files")

            # Tax already performs the augmentation in their script
            # so there is no need to perform it here
            csv_file, csv_path = convert_xes_to_csv(xes, "./tmp")

            _ , _, val, _, train_val_path = split_train_val_test(csv_path, "./tmp", XES_Fields.CASE_COLUMN, do_train_val=True)
            # Select the fields for the declare miner files.
            # This avoids nan error importing in prom.
            select_columns(
                val,
                input_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN, XES_Fields.TIMESTAMP_COLUMN, XES_Fields.LIFECYCLE_COLUMN],
                category_columns=None,
                timestamp_format=Timestamp_Formats.TIMESTAMP_FORMAT_YMDHMS_DASH,
                output_columns=None, francescomarino_fix=True
            )
            xes_file, xes_path = convert_csv_to_xes(train_val_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            xes_file_val, xes_path_val = convert_csv_to_xes(val, "./tmp", EXTENSIONS.XES_COMPRESSED)
            move_files(xes_path, "Process-Sequence-Prediction-with-A-priori-knowledge/data/declare_miner_files", EXTENSIONS.XES_COMPRESSED)
            move_files(xes_path_val, "Process-Sequence-Prediction-with-A-priori-knowledge/data/declare_miner_files", EXTENSIONS.XES_COMPRESSED)

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
                output_columns=output_columns, categorize=True, francescomarino_fix=True
            )
            csv_path, train_path, val_path, test_path = split_train_val_test(csv_path, "./tmp", "CaseID")
            move_files(csv_path, "Process-Sequence-Prediction-with-A-priori-knowledge/data", EXTENSIONS.CSV)

    elif arguments.net == "hinkka":
        for xes in dataset_list:
            print("Process: ", xes)
            csv_file, csv_path = convert_xes_to_csv(xes, "./tmp")
            attributes = load_attributes_from_file("attributes.yaml", Path(xes).name)

            output_columns = {
                XES_Fields.CASE_COLUMN: "CaseID",
                XES_Fields.ACTIVITY_COLUMN: "ActivityID",
                XES_Fields.TIMESTAMP_COLUMN: "CompleteTimestamp"
            }
            select_columns(
                csv_path,
                input_columns=[XES_Fields.CASE_COLUMN, XES_Fields.ACTIVITY_COLUMN, XES_Fields.TIMESTAMP_COLUMN] + attributes,
                category_columns=[XES_Fields.CASE_COLUMN],
                timestamp_format=Timestamp_Formats.TIMESTAMP_FORMAT_YMDHMS_DASH,
                output_columns=None, categorize=True
            )

            make_dir_if_not_exists("hinkka/src/testdata")
            make_dir_if_not_exists("hinkka/src/output")
            json_path = convert_csv_to_json(csv_path, "hinkka/src/testdata", attributes, Timestamp_Formats.TIMESTAMP_FORMAT_YMDHMS_DASH, prettify=True)
            csv_path, train_path, val_path, test_path = split_train_val_test(csv_path, "./tmp", XES_Fields.CASE_COLUMN)

            move_files(csv_path, "hinkka/src/testdata", EXTENSIONS.CSV)

    elif arguments.net == "rama":
        for xes in dataset_list:
            print("Process: ", xes)
            make_dir_if_not_exists("rama")
            csv_file, csv_path = convert_xes_to_csv(xes, "./tmp")
            csv_path, train_path, val_path, test_path, train_val_path = split_train_val_test(csv_path, "./tmp", XES_Fields.CASE_COLUMN, do_train_val=True)
            xes_file, xes_path = convert_csv_to_xes(csv_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            train_file, train_path = convert_csv_to_xes(train_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            val_file, val_path = convert_csv_to_xes(val_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            test_file, test_path = convert_csv_to_xes(test_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            train_val_file, train_val_path = convert_csv_to_xes(train_val_path, "./tmp", EXTENSIONS.XES_COMPRESSED)
            move_files(xes_path, "rama", EXTENSIONS.XES_COMPRESSED)

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
