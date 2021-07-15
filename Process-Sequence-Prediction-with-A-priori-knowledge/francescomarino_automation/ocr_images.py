# python -m pip install opencv-python pytesseract pandas
# sudo apt install libtesseract-dev tesseract-ocr
# Assume images 1698x859
import cv2
import pytesseract
import numpy as np
import os
import tqdm
import json
import re
from difflib import SequenceMatcher
import pandas as pd

# OPENCV MISC FUNCTIONS

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

start_X = 591
start_Y = 258
height_rule = 276-start_Y
width_rule = 1326-start_X

skip_Y = 315-start_Y

start_Y_activation = 284
height_activation = height_rule
width_activation = 691 - start_X

start_Y_fulfilments = start_Y_activation
start_X_fulfilments = 695
height_fulfilments = height_rule
width_fulfilments = 789 - start_X_fulfilments


def preprocess_image(image):
    # Sikuli screenshots are 98 DPIs and we require 400 dpi. A factor of 4 gives the best results
    image = cv2.resize(image, (0,0), fx = 4, fy = 4)
    image = get_grayscale(image)
    image = remove_noise(image)
    return image


# If the data file with the text extracted from the images does not exist
# Proceed to apply ocr on the extracted screenshots
if not os.path.isfile("raw_rules.json"):
    raw_rules = []
    progress_bar = tqdm.tqdm(total=len(os.listdir("declare_rules")) * 13)
    for i, image in enumerate(os.listdir("declare_rules")):
        img = cv2.imread(os.path.join("declare_rules", image))
        Y_rule = start_Y
        X_rule = start_X
        Y_activation = start_Y_activation
        X_activation = start_X
        Y_fulfilment = start_Y_fulfilments
        X_fulfilment = start_X_fulfilments
        for j in range(13):
            rule = img[Y_rule:Y_rule + height_rule , X_rule:X_rule + width_rule]
            activation = img[Y_activation: Y_activation + height_activation, X_activation:X_activation + width_activation]
            fulfilment = img[Y_fulfilment : Y_fulfilment + height_fulfilments, X_fulfilment : X_fulfilment + width_fulfilments]
            Y_rule += skip_Y
            Y_activation += skip_Y
            Y_fulfilment += skip_Y
            
            rule = preprocess_image(rule)
            activation = preprocess_image(activation)
            fulfilment = preprocess_image(fulfilment)
            rule_str = pytesseract.image_to_string(rule)
            activation_str = pytesseract.image_to_string(activation)
            fulfilment_str = pytesseract.image_to_string(fulfilment)

            raw_rules.append(
                {
                    "rule" : rule_str,
                    "activation_str" : activation_str,
                    "fulfilment_str" : fulfilment_str,
                    "image" : image
                }
            )
            progress_bar.update(1)

    with open("raw_rules.json", "w") as json_f:
        json.dump(raw_rules, json_f)

# Get similarity between two strings
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Load the raw rule data and activity assignment data
with open("raw_rules.json", "r") as json_f:
    raw_rules = json.load(json_f)

activity_assignments = {}
for log in os.listdir("../data/activity_assignments"):
    with open(os.path.join("../data/activity_assignments", log), "r") as log_f:
        line = log_f.readline().replace("'", "\"")
        assignments = json.loads(line)
        activity_assignments[log.replace(".csv", "").lower()] = assignments

existence_regex = "existence: \[(.*?)\].*"
fold_regex = "fold\\d_variation\\d_(.*)"
response_regex= "response: \[(.*?)\].*?\[(.*?)\].*"
activation_regex = ".*:.*?(\d+).*"


# Sanitize the activity by comparing its similarity to the ground truth activities of the log
def sanitize_activity(dirty_activity, act_assignments, log):
    assignments_for_log = act_assignments[log]
    activities = list(assignments_for_log.keys())
    if dirty_activity not in assignments_for_log:
        similarities = [similar(dirty_activity, activity) for activity in activities]
        # Find the activity with the most similarity to the given one
        resolved_activity = activities[similarities.index(max(similarities))]
        print("Resolving ", dirty_activity, " to ", resolved_activity)
        return resolved_activity
    else:
        return dirty_activity

# Sanitize the rules
rules_sanitized = []
for rule_info in raw_rules:
    rule_sanitized = {}
    raw_rule = rule_info["rule"]
    fold = rule_info["image"].replace(".png", "").lower()
    log = re.match(fold_regex, fold).group(1)
    rule_sanitized["fold"] = fold
    # Parse the rule type
    if "existence:" in raw_rule:
        rule_type = "existence"
        rule_sanitized["type"] = rule_type
    elif "response:" in raw_rule:
        rule_type = "response"
        rule_sanitized["type"] = "response"
    else:
        print("DISCARDING: ", raw_rule)
        continue

    # Parse the activities forming the rule
    if rule_type == "existence":
        matches = re.match(existence_regex, raw_rule)
        if matches is None:
            print("ERROR processing regex: ", existence_regex, " with rule line: ", raw_rule)
        else:
            activity_one = matches.group(1)
            activity_one = sanitize_activity(activity_one, activity_assignments, log)
            rule_sanitized["activity_one"] = activity_one
            rule_sanitized["assignment_one"] = int(activity_assignments[log][activity_one])
    else:
        matches = re.match(response_regex, raw_rule)
        if matches is None:
            print("ERROR processing regex: ", response_regex, " with rule line: ", raw_rule)
        else:
            activity_one = matches.group(1)
            activity_one = sanitize_activity(activity_one, activity_assignments, log)
            activity_two = matches.group(2)
            activity_two = sanitize_activity(activity_two, activity_assignments, log)
            rule_sanitized["activity_one"] = activity_one
            rule_sanitized["activity_two"] = activity_two
            rule_sanitized["assignment_one"] = int(activity_assignments[log][activity_one])
            rule_sanitized["assignment_two"] = int(activity_assignments[log][activity_two])

    # Parse the number of activations
    raw_activations = rule_info["activation_str"]
    n_activations = int(re.match(activation_regex, raw_activations).group(1))
    rule_sanitized["n_activations"] = n_activations

    # Parse the number of fulfilments
    raw_fulfilments = rule_info["fulfilment_str"]
    n_fulfilments = int(re.match(activation_regex, raw_fulfilments).group(1))
    rule_sanitized["n_fulfilments"] = n_fulfilments

    rules_sanitized.append(rule_sanitized)


rule_df = pd.DataFrame(rules_sanitized)

# Filter dataframe for the number of fulfilments/activations >= 0.5
rule_df = rule_df[(rule_df["n_fulfilments"] / rule_df["n_activations"]) >= 0.5]

rules = []
for i, group in rule_df.groupby("fold"):
    responses = group[group["type"] == "response"].sort_values("n_activations").head(3)
    #print("Best 3 responses", responses["assignment_two"])
    existence = group[group["type"] == "existence"]
    join = responses["assignment_one"].isin(existence["assignment_one"])
    true_responses = responses[join]
    rule_str_array = []
    for i, rule in true_responses.iterrows():
        rule_str = "[]( ( \\\"" + str(int(rule["assignment_one"])) + "\\\" ) -> <>( \\\"" + str(int(rule["assignment_two"])) +  "\\\" ) )  /\\\\ <>\\\"" + str(int(rule["assignment_one"])) + "\\\""
        rule_str_array.append(rule_str)

    rule_str = "  /\\\\ ".join(rule_str_array)
    rule_str = "\"" + rule_str + "\""
    full_str = "\"" + rule["fold"] + "\" : " + rule_str
    rules.append(full_str)

full_rules = "\n".join(rules)

with open("../src/formulas.yaml", "w") as formula_file:
    formula_file.write(full_rules)
         

