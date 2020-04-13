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
        # Move from tmp to the corresponding folder
        os.replace(csv_path, os.path.join("ImagePPMiner/dataset", csv_file))

elif arguments.net == "mauro":
    for xes in dataset_list:
        #csv_file, csv_path = augment_xes_end_activity_to_csv(xes, "./tmp")
        csv_file, csv_path = convert_xes_to_csv(xes, "./tmp")
        convert_csv_to_xes(csv_path, "./tmp")

else:
    print("Unrecognized approach")

delete_tmp()



