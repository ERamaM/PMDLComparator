import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow.keras.layers import Conv2D, Activation
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
import sys

seed = 123
np.random.seed(seed)
from tensorflow.compat.v1 import set_random_seed

set_random_seed(seed)

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--dataset", help="Raw dataset to prepare", required=True)
parser.add_argument("--train", help="Start training the neural network", action="store_true")
parser.add_argument("--test", help="Start testing next event of the neural network", action="store_true")
arguments = parser.parse_args()

if not (arguments.train or arguments.test):
    print("--train or --test (or both) are required")
    sys.exit(-3)

dataset_name = Path(arguments.dataset).stem.split(".")[0].lower()

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Create the model and output folders
import os

if not os.path.exists("results"):
    os.mkdir("results")
if not os.path.exists("models"):
    os.mkdir("models")


def get_label(act):
    i = 0
    list_label = []
    while i < len(act):
        j = 0
        while j < (len(act.iat[i, 0]) - 1):
            if j > 0:
                list_label.append(act.iat[i, 0][j + 1])
            else:
                pass
            j = j + 1
        i = i + 1
    return list_label


def dataset_summary(dataset):
    df = pd.read_csv(dataset, sep=",")
    print("Activity Distribution\n", df['Activity'].value_counts())
    n_caseid = df['CaseID'].nunique()
    n_activity = df['Activity'].nunique()
    print("Number of CaseID", n_caseid)
    print("Number of Unique Activities", n_activity)
    print("Number of Activities", df['Activity'].count())
    cont_trace = df['CaseID'].value_counts(dropna=False)
    max_trace = max(cont_trace)
    print("Max lenght trace", max_trace)
    print("Mean lenght trace", np.mean(cont_trace))
    print("Min lenght trace", min(cont_trace))
    return df, max_trace, n_caseid, n_activity


def get_image(act_val, time_val, max_trace, n_activity):
    i = 0
    matrix_zero = [max_trace, n_activity, 2]
    image = np.zeros(matrix_zero)
    list_image = []
    while i < len(time_val):
        j = 0
        list_act = []
        list_temp = []
        """
        cont1, cont2, cont3, cont4, cont5, cont6 = 0, 0, 0, 0, 0, 0
        diff1, diff2, diff3, diff4, diff5, diff6 = 0, 0, 0, 0, 0, 0
        while j < (len(act_val.iat[i, 0]) - 1):
            start_trace = time_val.iat[i, 0][0]
            if act_val.iat[i, 0][0 + j] == 1:
                cont1 = cont1 + 1
                diff1 = time_val.iat[i, 0][0 + j] - start_trace
            elif act_val.iat[i, 0][0 + j] == 2:
                cont2 = cont2 + 1
                diff2 = time_val.iat[i, 0][0 + j] - start_trace
            elif act_val.iat[i, 0][0 + j] == 3:
                cont3 = cont3 + 1
                diff3 = time_val.iat[i, 0][0 + j] - start_trace
            elif act_val.iat[i, 0][0 + j] == 4:
                cont4 = cont4 + 1
                diff4 = time_val.iat[i, 0][0 + j] - start_trace
            elif act_val.iat[i, 0][0 + j] == 5:
                cont5 = cont5 + 1
                diff5 = time_val.iat[i, 0][0 + j] - start_trace
            elif act_val.iat[i, 0][0 + j] == 6:
                cont6 = cont6 + 1
                diff6 = time_val.iat[i, 0][0 + j] - start_trace

            list_act.append([cont1, cont2, cont3, cont4, cont5, cont6])
            list_temp.append([diff1, diff2, diff3, diff4, diff5, diff6])
            j = j + 1
            cont = 0
            lenk = len(list_act) - 1
            while cont <= lenk:
                image[(max_trace - 1) - cont][0] = [list_act[lenk - cont][0], list_temp[lenk - cont][0]]
                image[(max_trace - 1) - cont][1] = [list_act[lenk - cont][1], list_temp[lenk - cont][1]]
                image[(max_trace - 1) - cont][2] = [list_act[lenk - cont][2], list_temp[lenk - cont][2]]
                image[(max_trace - 1) - cont][3] = [list_act[lenk - cont][3], list_temp[lenk - cont][3]]
                image[(max_trace - 1) - cont][4] = [list_act[lenk - cont][4], list_temp[lenk - cont][4]]
                image[(max_trace - 1) - cont][5] = [list_act[lenk - cont][5], list_temp[lenk - cont][5]]
                cont = cont + 1
            if cont == 1:
                pass
            else:
                list_image.append(image)
                image = np.zeros(matrix_zero)
        i = i + 1
        """
        cont = [0] * n_activity
        diff = [0] * n_activity
        while j < (len(act_val.iat[i, 0]) - 1):
            start_trace = time_val.iat[i, 0][0]
            label = act_val.iat[i, 0][0 + j]
            cont[label - 1] = cont[label - 1] + 1
            diff[label - 1] = time_val.iat[i, 0][0 + j] - start_trace

            # If we don't copy we would have the same last array in the whole image
            list_act.append(cont.copy())
            list_temp.append(diff.copy())
            j = j + 1
            count = 0
            lenk = len(list_act) - 1
            while count <= lenk:
                for k in range(n_activity):
                    image[(max_trace - 1) - count][k] = [list_act[lenk - count][k], list_temp[lenk - count][k]]
                count = count + 1
            # Skip prefix of length 1
            if count == 1:
                pass
            else:
                list_image.append(image)
                image = np.zeros(matrix_zero)
        i = i + 1
    return list_image


# import dataset
df, max_trace, n_caseid, n_activity = dataset_summary(arguments.dataset)

# group by activity and timestamp by caseid
act = df.groupby('CaseID').agg({'Activity': lambda x: list(x)})
temp = df.groupby('CaseID').agg({'Timestamp': lambda x: list(x)})

# Load the splits
# Perform the same operations as above
dataset_directory = Path(arguments.dataset).parent
dataset_filename = Path(arguments.dataset).stem
df_train, _, _, _ = dataset_summary(os.path.join(dataset_directory, "train_" + dataset_filename + ".csv"))
df_val, _, _, _ = dataset_summary(os.path.join(dataset_directory, "val_" + dataset_filename + ".csv"))

train_act = df_train.groupby('CaseID').agg({'Activity': lambda x: list(x)})
train_temp = df_train.groupby('CaseID').agg({'Timestamp': lambda x: list(x)})

val_act = df_val.groupby('CaseID').agg({'Activity': lambda x: list(x)})
val_temp = df_val.groupby('CaseID').agg({'Timestamp': lambda x: list(x)})


# generate training and test set
X_train = get_image(train_act, train_temp, max_trace, n_activity)
X_val = get_image(val_act, val_temp, max_trace, n_activity)

l_train = get_label(train_act)
l_val = get_label(val_act)
# Get the labels for the whole training set
# It may happen that some labels are present in the test set but no in
# the training (Helpdesk)
l_total = get_label(act)

le = preprocessing.LabelEncoder()
# Calculate the labels on the whole training set but transform
# only the splits
le.fit(l_total)
l_train = le.transform(l_train)
l_val = le.transform(l_val)
num_classes = le.classes_.size
print(list(le.classes_))

X_train = np.asarray(X_train)
l_train = np.asarray(l_train)

X_val = np.asarray(X_val)
l_val = np.asarray(l_val)


train_Y_one_hot = to_categorical(l_train, num_classes)
val_Y_one_hot = to_categorical(l_val, num_classes)


# define neural network architecture
model = Sequential()
reg = 0.0001
input_shape = (max_trace, n_activity, 2)
model.add(Conv2D(32, (2, 2), input_shape=input_shape, padding='same', kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(reg)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (4, 4), padding='same', kernel_regularizer=regularizers.l2(reg), ))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Respecting the paper, the BPI2012W_Complete removes this layer
# Seems that if we do not remove it it causes an exception of layer of dimension -1
# Only add the layer for helpdesk (the error happens in more logs).
if dataset_name == "helpdesk":
    model.add(Conv2D(128, (8, 8), padding='same', kernel_regularizer=regularizers.l2(reg), ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax', name='act_output'))

print(model.summary())

opt = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=opt, metrics=['accuracy'])

import math
BATCH_SIZE = 128

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return math.ceil(len(self.X) / BATCH_SIZE)

    def __getitem__(self, idx):
        X = self.X[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        y = self.y[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        return X, y


# history = model.fit(X_train, {'act_output': train_Y_one_hot}, validation_data=(X_val, val_Y_one_hot), verbose=1,
#                    callbacks=[early_stopping], batch_size=128, epochs=500)
if arguments.train:
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)
    history = model.fit_generator(DataGenerator(X_train, train_Y_one_hot), validation_data=(X_val, val_Y_one_hot),
                                verbose=1, epochs=500, callbacks=[early_stopping])

    model.save("models/" + dataset_name + ".h5")

# Load the test part
if arguments.test:
    results_file = open("results/" + dataset_name + ".txt", mode="w")
    raw_results_file = open("results/raw_" + dataset_name + ".txt", mode="w")

    model.load_weights("models/" + dataset_name + ".h5")
    # Print confusion matrix for training data
    y_pred_train = model.predict(X_train)
    # Take the class with the highest probability from the train predictions
    max_y_pred_train = np.argmax(y_pred_train, axis=1)
    print(classification_report(l_train, max_y_pred_train, digits=3))

    df_test, _, _, _ = dataset_summary(os.path.join(dataset_directory, "test_" + dataset_filename + ".csv"))
    test_act = df_test.groupby('CaseID').agg({'Activity': lambda x: list(x)})
    test_temp = df_test.groupby('CaseID').agg({'Timestamp': lambda x: list(x)})
    X_test = get_image(test_act, test_temp, max_trace, n_activity)
    l_test = get_label(test_act)
    l_test = le.transform(l_test)
    X_test = np.asarray(X_test)
    l_test = np.asarray(l_test)
    test_Y_one_hot = to_categorical(l_test, num_classes)
    score = model.evaluate(X_test, test_Y_one_hot, verbose=1, batch_size=1)

    results_file.write('\nLoss on test data: ' + str(score[0]) + "\n")

    y_pred_test = model.predict(X_test)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(test_Y_one_hot, axis=1)
    results_file.write(classification_report(max_y_test, max_y_pred_test, digits=3))
    from sklearn.metrics import brier_score_loss, matthews_corrcoef


    def calculate_brier_score(y_pred, y_true):
    # From: https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))


    results_file.write("\nBrier score: " + str(calculate_brier_score(y_pred_test, test_Y_one_hot)))
    from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score

    mcc = matthews_corrcoef(max_y_test, max_y_pred_test)
    precision = precision_score(max_y_test, max_y_pred_test, average="weighted")
    recall = recall_score(max_y_test, max_y_pred_test, average="weighted")
    f1 = f1_score(max_y_test, max_y_pred_test, average="weighted")
    accuracy = accuracy_score(max_y_test, max_y_pred_test)
    results_file.write("\nAccuracy: " + str(accuracy))
    results_file.write("\nMCC: " + str(mcc))
    results_file.write("\nWeighted Precision: " + str(precision))
    results_file.write("\nWeighted Recall: " + str(recall))
    results_file.write("\nWeighted F1: " + str(f1))

    raw_results_file.write("prefix_length;ground_truth;predicted;prediction_probs\n")
    for X, y, y_pred, probs in zip(X_test, max_y_test, max_y_pred_test, y_pred_test):
        # First reduce the images to a sum of the accumulated activities.
        # Get the last accumulated activity and, then, the number of activities of the prefix.
        length = int(np.sum(X, axis=1)[-1][0]) - 1
        #length = np.sum(np.sum(X, axis=-1).astype("bool").astype("int"))
        raw_results_file.write(str(length) + ";" + str(y) + ";" + str(y_pred) + ";" + str(np.array2string(probs,separator=",", max_line_width=99999)) + "\n")

    results_file.close()
    raw_results_file.close()
