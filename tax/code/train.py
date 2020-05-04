'''
this script trains an LSTM model on one of the data files in the data folder of
this repository. the input file can be changed to another file from the data folder
by changing its name in line 46.

it is recommended to run this script on GPU, as recurrent networks are quite 
computationally intensive.

Author: Niek Tax
'''

from __future__ import print_function, division
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import get_file
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from collections import Counter
import numpy as np
import random
import sys
import os
import copy
import csv
import time
from datetime import datetime
from math import log

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

tf.compat.v1.set_random_seed(42)
# tf.enable_eager_execution()
import random
random.seed(42)
np.random.seed(42)

import argparse

parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--train", help="Start the training of the neural network", action="store_true")
parser.add_argument("--test", help="Start the testing of next event", action="store_true")
args = parser.parse_args()

if not (args.train or args.test):
    print("Argument --train or --test (or both) are required")
    sys.exit(-3)

eventlog = args.dataset


def load_file(eventlog):
    ascii_offset = 161
    csvfile = open('%s' % eventlog, 'r')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader, None)  # skip the headers
    lastcase = ''
    line = ''
    firstLine = True
    lines = []
    timeseqs = []
    timeseqs2 = []
    timeseqs3 = []
    timeseqs4 = []
    times = []
    times2 = []
    times3 = []
    times4 = []
    numlines = 0
    casestarttime = None
    lasteventtime = None
    for row in spamreader:
        t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
        if row[0] != lastcase:
            casestarttime = t
            lasteventtime = t
            lastcase = row[0]
            if not firstLine:
                lines.append(line)
                timeseqs.append(times)
                timeseqs2.append(times2)
                timeseqs3.append(times3)
                timeseqs4.append(times4)
            line = ''
            times = []
            times2 = []
            times3 = []
            times4 = []
            numlines += 1
        line += chr(int(row[1]) + ascii_offset)
        timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(lasteventtime))
        timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(casestarttime))
        midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = datetime.fromtimestamp(time.mktime(t)) - midnight
        timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
        timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
        timediff3 = timesincemidnight.seconds
        timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday()
        times.append(timediff)
        times2.append(timediff2)
        times3.append(timediff3)
        times4.append(timediff4)
        lasteventtime = t
        firstLine = False
    # add last case
    lines.append(line)
    timeseqs.append(times)
    timeseqs2.append(times2)
    timeseqs3.append(times3)
    timeseqs4.append(times4)
    numlines += 1
    return lines, timeseqs, timeseqs2, timeseqs3, timeseqs4

from pathlib import Path
import os
eventlog_name = Path(eventlog).stem
extension = ".csv"
folders = Path(eventlog).parent

lines, timeseqs, timeseqs2, timeseqs3, timeseqs4 = load_file(eventlog)
lines_train, timeseqs_train, timeseqs2_train, timeseqs3_train, timeseqs4_train = load_file(os.path.join(folders, "train_" + eventlog_name + extension))
lines_val, timeseqs_val, timeseqs2_val, timeseqs3_val, timeseqs4_val = load_file(os.path.join(folders, "val_" + eventlog_name + extension))
lines_test, timeseqs_test, timeseqs2_test, timeseqs3_test, timeseqs4_test = load_file(os.path.join(folders, "test_" + eventlog_name + extension))

# The divisors are calculated from only the training set
# Otherwise we would be filtering information from the test set
# into the training set
divisor = np.mean([item for sublist in timeseqs_train for item in sublist])  # average time between events
print('divisor: {}'.format(divisor))
divisor2 = np.mean(
    [item for sublist in timeseqs2_train for item in sublist])  # average time between current and first events
print('divisor2: {}'.format(divisor2))

# Add the termination character AFTER calculating the metrics
lines = list(map(lambda x: x + '!', lines))

# We need to vectorize the dataset at once because the folds could
# contain activities that are not in the other folds
maxlen = max(map(lambda x: len(x), lines))  # find maximum line size
chars = map(lambda x: set(x), lines)
chars = list(set().union(*chars))
chars.sort()
target_chars = copy.copy(chars)
chars.remove('!')
num_features = len(chars) + 5
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))


def vectorize_fold(fold1, fold1_t, fold1_t2, fold1_t3, fold1_t4, divisor, divisor2):
    lines = fold1
    lines_t = fold1_t
    lines_t2 = fold1_t2
    lines_t3 = fold1_t3
    lines_t4 = fold1_t4
    step = 1
    sentences = []
    softness = 0
    next_chars = []
    sentences_t = []
    sentences_t2 = []
    sentences_t3 = []
    sentences_t4 = []
    next_chars_t = []
    next_chars_t2 = []
    next_chars_t3 = []
    next_chars_t4 = []
    # Construct the prefixes and store them in sentences, sentences_t, ...
    for line, line_t, line_t2, line_t3, line_t4 in zip(lines, lines_t, lines_t2, lines_t3, lines_t4):
        for i in range(0, len(line), step):
            # This would be an empty prefix, and it doesn't make much sense to predict based on nothing
            if i == 0:
                continue
            sentences.append(line[0: i])
            sentences_t.append(line_t[0:i])
            sentences_t2.append(line_t2[0:i])
            sentences_t3.append(line_t3[0:i])
            sentences_t4.append(line_t4[0:i])
            # Store the desired prediction
            next_chars.append(line[i])
            if i == len(line) - 1:  # special case to deal time of end character
                next_chars_t.append(0)
                next_chars_t2.append(0)
                next_chars_t3.append(0)
                next_chars_t4.append(0)
            else:
                next_chars_t.append(line_t[i])
                next_chars_t2.append(line_t2[i])
                next_chars_t3.append(line_t3[i])
                next_chars_t4.append(line_t4[i])

    print('nb sequences:', len(sentences))
    print('Vectorization...')
    # Matrix containing the training data
    X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
    # Target event prediction data
    y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
    # Target time prediction data
    y_t = np.zeros((len(sentences)), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        leftpad = maxlen - len(sentence)
        next_t = next_chars_t[i]
        sentence_t = sentences_t[i]
        sentence_t2 = sentences_t2[i]
        sentence_t3 = sentences_t3[i]
        sentence_t4 = sentences_t4[i]
        for t, char in enumerate(sentence):
            multiset_abstraction = Counter(sentence[:t + 1])
            for c in chars:
                if c == char:
                    X[i, t + leftpad, char_indices[c]] = 1
            X[i, t + leftpad, len(chars)] = t + 1
            X[i, t + leftpad, len(chars) + 1] = sentence_t[t] / divisor
            X[i, t + leftpad, len(chars) + 2] = sentence_t2[t] / divisor2
            X[i, t + leftpad, len(chars) + 3] = sentence_t3[t] / 86400
            X[i, t + leftpad, len(chars) + 4] = sentence_t4[t] / 7
        for c in target_chars:
            if c == next_chars[i]:
                y_a[i, target_char_indices[c]] = 1 - softness
            else:
                y_a[i, target_char_indices[c]] = softness / (len(target_chars) - 1)
        y_t[i] = next_t / divisor
        np.set_printoptions(threshold=sys.maxsize)
    return X, target_char_indices, y_a, y_t

X, target_char_indices, y_a, y_t = vectorize_fold(lines, timeseqs, timeseqs2, timeseqs3, timeseqs4, divisor, divisor2)
X_train, _, y_a_train, y_t_train = vectorize_fold(lines_train, timeseqs_train, timeseqs2_train, timeseqs3_train, timeseqs4_train, divisor, divisor2)
X_val, _, y_a_val, y_t_val = vectorize_fold(lines_val, timeseqs_val, timeseqs2_val, timeseqs3_val, timeseqs4_val, divisor, divisor2)
X_test, _, y_a_test, y_t_test = vectorize_fold(lines_test, timeseqs_test, timeseqs2_test, timeseqs3_test, timeseqs4_test, divisor, divisor2)
# Split in chunks



# build the model:
print('Build model...')
main_input = Input(shape=(maxlen, num_features), name='main_input')
# train a 2-layer LSTM with one shared layer
l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(
    main_input)  # the shared layer
b1 = BatchNormalization()(l1)
l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(
    b1)  # the layer specialized in activity prediction
b2_1 = BatchNormalization()(l2_1)
l2_2 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(
    b1)  # the layer specialized in time prediction
b2_2 = BatchNormalization()(l2_2)
act_output = Dense(len(target_chars), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(
    b2_1)
time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)

model = Model(inputs=[main_input], outputs=[act_output, time_output])

opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

import distutils.dir_util
import os

distutils.dir_util.mkpath("output_files/models/" + eventlog)

model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'}, optimizer=opt, metrics={"act_output" : "acc", "time_output" : "mae"})
early_stopping = EarlyStopping(monitor='val_loss', patience=42)
best_model = "models/" + eventlog_name + ".h5"
model_checkpoint = ModelCheckpoint(best_model, monitor='val_loss', verbose=1,
                                   save_best_only=True, save_weights_only=False, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto', min_delta=0.0001,
                               cooldown=0, min_lr=0)

model.summary()
# We can't use validation split since that split would do a split of "events" and not a split of "traces"
# We need to manually set the valiation set
if args.train:
    model.fit(X_train, {'act_output': y_a_train, 'time_output': y_t_train}, validation_data = (X_val, {"act_output" : y_a_val, "time_output" : y_t_val}),verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=maxlen, epochs=200)


if args.test:
    model.load_weights(best_model)
    model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'}, optimizer=opt, metrics={"act_output" : "acc", "time_output" : "mae"})
    metrics = model.evaluate(X_test, {'act_output': y_a_test, 'time_output': y_t_test}, verbose=1, batch_size=maxlen)

    y_a_pred_probs = model.predict([X_test])[0]
    y_a_pred = np.argmax(y_a_pred_probs, axis=1)
    y_true = np.argmax(y_a_test, axis=1)
    from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score

    def calculate_brier_score(y_pred, y_true):
        # From: https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
        return np.mean(np.sum((y_true - y_pred)**2, axis=1))
    with open("results/" + eventlog_name +"_next_event.log", "w") as file:
        for metric, name in zip(metrics, model.metrics_names):
            if name == "time_output_mae":
                # Undo the standarization done in the line y_t[i] = next_t / divisor
                # Divide the result by 86400 to have the result in days
                file.write("mae_in_days: " + str(metric * (divisor / 86400)) + "\n")
            else:
                file.write(str(name) + ": " + str(metric) + "\n")

        acc = accuracy_score(y_true, y_a_pred)
        mcc = matthews_corrcoef(y_true, y_a_pred)
        precision = precision_score(y_true, y_a_pred, average="weighted")
        recall = recall_score(y_true, y_a_pred, average="weighted")
        f1 = f1_score(y_true, y_a_pred, average="weighted")
        brier_score = calculate_brier_score(y_a_pred_probs, y_a_test)
        file.write("\nACC Sklearn: " + str(acc))
        file.write("\nMCC: " + str(mcc))
        file.write("\nBrier score: " + str(brier_score))
        file.write("\nWeighted Precision: " + str(precision))
        file.write("\nWeighted Recall: " + str(recall))
        file.write("\nWeighted F1: " + str(f1))

    with open(os.path.join("results", "raw_" + eventlog_name + ".txt"), "w") as raw_file:
        raw_file.write("prefix_length;ground_truth;predicted;prediction_probs\n")
        for X, y_t, y_p, y_p_pred in zip(X_test, y_true, y_a_pred, y_a_pred_probs):
            raw_file.write(str(np.count_nonzero(np.sum(X, axis=-1))) + ";" + str(y_t) + ";" + str(y_p) + ";" + np.array2string(
                y_p_pred, separator=",", max_line_width=99999) + "\n")
