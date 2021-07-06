import os
import argparse
from io import StringIO
import subprocess
import pandas as pd

parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--approach", help="Approach to extract info for", required=True)
arguments = parser.parse_args()

approach = arguments.approach


def is_tool(name):
    try:
        devnull = open(os.devnull)
        subprocess.Popen([name], stdout=devnull, stderr=devnull).communicate()
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            return False
    return True


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

