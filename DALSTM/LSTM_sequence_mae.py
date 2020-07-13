"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import model_from_json
from math import sqrt
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import metrics
from load_dataset import load_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys, os
import tensorflow as tf

# Avoid saturating the GPU memory
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
tf.random.set_seed(42)
np.random.seed(42)

if len(sys.argv) < 3:
    sys.exit("python LSTM_sequence.py n_neurons n_layers dataset")
n_neurons = 150
n_layers = 2
import argparse
from pathlib import Path
parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--train", help="Start training the neural network", action="store_true")
parser.add_argument("--test", help="Start testing next event of the neural network", action="store_true")
args = parser.parse_args()
dataset = args.dataset
dataset_name = Path(dataset).name
dataset_directory = Path(dataset).parent

print("Dataset name: ", dataset_name)

# fix random seed for reproducibility
np.random.seed(42)
tf.compat.v1.set_random_seed(42)

(X, y), values = load_dataset(dataset)
(X_train, y_train), _ = load_dataset(os.path.join(dataset_directory, "train_" + dataset_name), values)
(X_val, y_val), _ = load_dataset(os.path.join(dataset_directory, "val_" + dataset_name), values)
(X_test, y_test), _ = load_dataset(os.path.join(dataset_directory, "test_" + dataset_name), values)


# normalize input data
# compute the normalization values only on training set
max = [0] * len(X_train[0][0])
for a1 in X_train:
    for s in a1:
        for i in range(len(s)):
            if s[i] > max[i]:
                max[i] = s[i]

print("MAX: ", max)

for a1 in X_train:
    for s in a1:
        for i in range(len(s)):
            if (max[i] > 0):  # alcuni valori hanno massimo a zero
                s[i] = s[i] / max[i]

for a1 in X_val:
    for s in a1:
        for i in range(len(s)):
            if (max[i] > 0):  # alcuni valori hanno massimo a zero
                s[i] = s[i] / max[i]

for a1 in X_test:
    for s in a1:
        for i in range(len(s)):
            if (max[i] > 0):  # alcuni valori hanno massimo a zero
                s[i] = s[i] / max[i]

X_train = np.asarray(X_train)
X_val = np.asarray(X_val)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)
y_test = np.asarray(y_test)

if dataset_name.lower() == "bpi_challenge_2013_incidents.csv":
    X_train = sequence.pad_sequences(X_train, dtype="int16")
    print("X_train: ", X_train[0])
    print("DEBUG: training shape", X_train.shape)
    maxlen = X_train.shape[1]
    X_test = sequence.pad_sequences(X_test, maxlen=X_train.shape[1], dtype="int16")
    X_val = sequence.pad_sequences(X_val, maxlen=X_train.shape[1], dtype="int16")
    print("DEBUG: test shape", X_test.shape)
else:
    X_train = sequence.pad_sequences(X_train)
    print("X_train: ", X_train[0])
    print("DEBUG: training shape", X_train.shape)
    maxlen = X_train.shape[1]
    X_test = sequence.pad_sequences(X_test, maxlen=X_train.shape[1])
    X_val = sequence.pad_sequences(X_val, maxlen=X_train.shape[1])
    print("DEBUG: test shape", X_test.shape)

# create the model
model = Sequential()
if n_layers == 1:
    model.add(
        LSTM(n_neurons, implementation=2, input_shape=(X_train.shape[1], X_train.shape[2]), recurrent_dropout=0.2))
    model.add(BatchNormalization())
else:
    for i in range(n_layers - 1):
        model.add(
            LSTM(n_neurons, implementation=2, input_shape=(X_train.shape[1], X_train.shape[2]), recurrent_dropout=0.2,
                 return_sequences=True))
        model.add(BatchNormalization())
    model.add(LSTM(n_neurons, implementation=2, recurrent_dropout=0.2))
    model.add(BatchNormalization())

# add output layer (regression)
model.add(Dense(1))
if not os.path.exists("model/model_data"):
    os.mkdir("model/model_data")

# compiling the model, creating the callbacks
model.compile(loss='mae', optimizer='Nadam', metrics=['mean_squared_error', 'mae', 'mape'])
print(model.summary())
early_stopping = EarlyStopping(patience=42)
model_checkpoint = ModelCheckpoint(
    "model/model_" + dataset + "_" + str(n_neurons) + "_" + str(n_layers) + "_weights_best.h5", monitor='val_loss',
    verbose=0, save_best_only=True,
    mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001,
                               cooldown=0, min_lr=0)


# train the model
if args.train:
    model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint, lr_reducer], epochs=500,
              batch_size=maxlen, verbose=1)
    # saving model to file
    model.save_weights(os.path.join("model", dataset_name + ".h5"))

if args.test:
    # Final evaluation of the model
    # Create new model with saved architecture
    # TODO it is possible to use the same model used for training, just loading the weights
    testmodel = model
    # load saved weigths to the test model
    testmodel.load_weights(os.path.join("model", dataset_name + ".h5"))
    # Compile model (required to make predictions)
    testmodel.compile(loss='mae', optimizer='Nadam', metrics=['mean_squared_error', 'mae', 'mape'])
    print("Created model and loaded weights from file")

    # compute metrics on test set (same order as in the metrics list given to the compile method)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Root Mean Squared Error: %.4f d MAE: %.4f d MAPE: %.4f%%" % (
    sqrt(scores[1] / ((24.0 * 3600) ** 2)), scores[2] / (24.0 * 3600), scores[3]))

    if not os.path.exists("results"):
        os.mkdir("results")

    with open("results/" + dataset_name + ".txt", "w") as result_file:
        result_file.write("Root Mean Squared Error: %.4f d MAE: %.4f d MAPE: %.4f%%" % (
            sqrt(scores[1] / ((24.0 * 3600) ** 2)), scores[2] / (24.0 * 3600), scores[3]))

