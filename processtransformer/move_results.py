import os
import shutil

for result_dir in os.listdir("./results"):
    fold = result_dir.replace(".csv", "")
    shutil.move("./results/" + result_dir + "/results_next_activity.csv", "./results/" + fold + "_results.txt")
    os.rmdir("./results/" + result_dir)
