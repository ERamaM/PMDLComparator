'''
this script trains an LSTM model on one of the data files in the data folder of
this repository. the input file can be changed to another file from the data folder
by changing its name in line 46.

it is recommended to run this script on GPU, as recurrent networks are quite 
computationally intensive.

Author: Niek Tax
'''

from __future__ import print_function, division
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input
from keras.utils.data_utils import get_file
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from collections import Counter
import unicodecsv
import numpy as np
import random
import sys
import os
import copy
import csv
import time
from itertools import izip
from datetime import datetime
from math import log

from keras.backend import set_session
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config=config)
tf.debugging.set_log_device_placement(True)
set_session(sess)  # set this TensorFlow session as the default session for Keras

tf.compat.v1.set_random_seed(42)
# tf.enable_eager_execution()
import random
random.seed(42)
np.random.seed(42)

import argparse

parser = argparse.ArgumentParser(description="SeqPred")
parser.add_argument("eventlog", type=str)
args = parser.parse_args()
eventlog = args.eventlog


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


train = int(round(len(lines) * 0.64))
val = int(round(len(lines) * 0.8))

print("Len lines: ", len(lines))

# The folds are a list of traces
fold1 = lines[:train]
fold1_t = timeseqs[:train]
fold1_t2 = timeseqs2[:train]
# The divisors are calculated from only the training set
# Otherwise we would be filtering information from the test set
# into the training set
divisor = np.mean([item for sublist in fold1_t for item in sublist])  # average time between events
print('divisor: {}'.format(divisor))
divisor2 = np.mean(
    [item for sublist in fold1_t2 for item in sublist])  # average time between current and first events
print('divisor2: {}'.format(divisor2))

fold2 = lines[train:val]

fold3 = lines[val:]

train_events = len("".join(fold1))
val_events = len("".join(fold2))
test_events = len("".join(fold3))

print("Train events: ", train_events)
print("Val events", val_events)
print("Test events: ", test_events)

# Add the termination character AFTER calculating the metrics
lines = list(map(lambda x: x + '!', lines))

# We need to vectorize the dataset at once because the folds could
# contain activities that are not in the other folds
fold1 = lines
fold1_t = timeseqs
fold1_t2 = timeseqs2
fold1_t3 = timeseqs3
fold1_t4 = timeseqs4
maxlen = max(map(lambda x: len(x), lines))  # find maximum line size
chars = map(lambda x: set(x), lines)
chars = list(set().union(*chars))
chars.sort()
target_chars = copy.copy(chars)
chars.remove('!')
num_features = len(chars) + 5


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
    # next lines here to get all possible characters for events and annotate them with numbers
    chars = map(lambda x: set(x), lines)
    chars = list(set().union(*chars))
    chars.sort()
    target_chars = copy.copy(chars)
    chars.remove('!')
    maxlen = max(map(lambda x: len(x), lines))  # find maximum line size
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
    target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
    num_features = len(chars) + 5
    print('num features: {}'.format(num_features))
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

X, target_char_indices, y_a, y_t = vectorize_fold(fold1, fold1_t, fold1_t2, fold1_t3, fold1_t4, divisor, divisor2)
X_train = X[:train_events]
X_validation = X[train_events: train_events+val_events]
X_test = X[train_events+val_events:]
y_train_a = y_a[:train_events]
y_validation_a = y_a[train_events: train_events+val_events]
y_test_a = y_a[train_events+val_events:]
y_train_t = y_t[:train_events]
y_validation_t = y_t[train_events: train_events+val_events]
y_test_t = y_t[train_events+val_events:]
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

model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'}, optimizer=opt, metrics=["acc"])
early_stopping = EarlyStopping(monitor='val_loss', patience=42)
model_checkpoint = ModelCheckpoint('output_files/models/' + eventlog + '/best_model.h5', monitor='val_loss', verbose=1,
                                   save_best_only=True, save_weights_only=False, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto', min_delta=0.0001,
                               cooldown=0, min_lr=0)

model.summary()
# We can't use validation split since that split would do a split of "events" and not a split of "traces"
# We need to manually set the valiation set
model.fit(X_train, {'act_output': y_train_a, 'time_output': y_train_t}, validation_data = (X_validation, {"act_output" : y_validation_a, "time_output" : y_validation_t}),verbose=1,
          callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=maxlen, epochs=200)


best_model = "output_files/models/" + eventlog + "/best_model.h5"
model.load_weights(best_model)
model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'}, optimizer=opt, metrics=["acc"])
metrics = model.evaluate(X_test, {'act_output': y_test_a, 'time_output': y_test_t}, verbose=1, batch_size=maxlen)
for metric, name in izip(metrics, model.metrics_names):
    print(name, ": ", metric)
