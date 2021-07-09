import os
import argparse
from io import StringIO
import subprocess
import pandas as pd
import re

parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--approach", help="Approach to extract info for", required=True)
parser.add_argument("--ts", help="Enable task spooler querying", action="store_true")
arguments = parser.parse_args()

approach = arguments.approach
ts = arguments.ts

log_regex = "fold(\\d)_variation(\\d)_(.*)"


def is_tool(name):
    try:
        devnull = open(os.devnull)
        subprocess.Popen([name], stdout=devnull, stderr=devnull).communicate()
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            return False
    return True


if ts:
    if is_tool("ts"):
        TS_EXECUTABLE = "ts"
    else:
        TS_EXECUTABLE = "tsp"

    # https://stackoverflow.com/questions/42201691/reading-a-complicated-table-with-pandas-task-spooler
    ts_command = "TS_SOCKET=/tmp/" + approach + " " + TS_EXECUTABLE + " | sed -e 's/\s\{2,12\}/,/g' | sed -e 's/\([[:digit:]].[[:digit:]]\{2\}\) /\1,/' "
    csv_str = subprocess.check_output(ts_command, shell=True)
    csv_str = csv_str.decode("utf-8")
    df_ts = pd.read_csv(StringIO(csv_str), sep=",")
    print("TS DF: ", df_ts)
    max_id = df_ts["ID"].max()
    min_id = df_ts["ID"].min()
    if not os.path.exists("./" + approach):
        os.mkdir("./" + approach)
    for i in range(min_id, max_id):
        ts_time_command = "TS_SOCKET=/tmp/" + approach + " " + TS_EXECUTABLE + " -i " + str(i)
        time_str = subprocess.check_output(ts_time_command, shell=True)
        if not os.path.isfile("./" + approach + "/" + str(i)):
            with open("./" + approach + "/" + str(i), "w") as f:
                f.write(time_str.decode("utf-8"))

information = []
for approach in os.listdir("."):
    if os.path.isfile(approach):
        continue
    for file in os.listdir("./" + approach):
        with open("./" + approach + "/" + file, "r") as f:
            contents = f.readlines()
            for line in contents:
                if re.match("Command: .*", line):
                    command_line = line
                if re.match("Time run: .*", line):
                    time_line = line
            timesec = re.match("Time run: (.*)s", time_line).groups()[0]
            matches = re.match(".* --fold_dataset (.*?) .*", command_line)
            if matches is not None:
                fold_log = matches.groups()[0].split("/")[-1]
            else:
                fold_log = re.match(".* --log (.*?) .*", command_line).groups[0].split("/")[-1]
            fold, variation, log = re.match(log_regex, fold_log).groups()
            log = log.replace(".csv", "").replace(".json", "").replace(".xes.gz", "").lower()
            information.append({"approach" : approach, "fold" : int(fold), "variation" : int(variation), "log" : log, "time" : float(timesec)})

data_df = pd.DataFrame(information)
print(data_df)
mean_df = data_df[["approach", "log", "time"]].groupby(["log", "approach"]).mean()
mean_df = mean_df.reset_index()
print("Result df: ", mean_df)
mean_df = mean_df.pivot(index="approach", columns="log", values="time")
mean_df.to_csv("mean_times.csv")

std_df = data_df[["approach", "log", "time"]].groupby(["log", "approach"]).std()
std_df = std_df.reset_index()
print("Result df: ", std_df)
std_df = std_df.pivot(index="approach", columns="log", values="time")
std_df.to_csv("std_times.csv")
