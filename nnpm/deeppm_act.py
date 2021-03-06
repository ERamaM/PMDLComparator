import numpy as np

seed = 123
np.random.seed(seed)
from tensorflow.compat.v1 import set_random_seed

set_random_seed(seed)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, \
    MaxPooling1D, Flatten, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils import load_data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
import sys
import csv
from time import perf_counter
import time

from tensorflow.keras.utils import Sequence

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


class DataGenerator(Sequence):
    def __init__(self, features, labels, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.labels = labels
        self.X_a = features[0]
        self.X_t = features[1]
        self.y_a = labels[0]
        self.y_t = labels[1]
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of steps per epoch'
        return int(np.floor(self.X_a.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.X_a.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X_a = np.empty((self.batch_size, self.X_a.shape[1]))
        X_t = np.empty((self.batch_size, self.X_t.shape[1]))
        y_a = np.empty((self.batch_size, self.y_a.shape[1]), dtype=int)
        y_t = np.empty((self.batch_size))

        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample
            X_a[i] = self.X_a[ID]
            X_t[i] = self.X_t[ID]

            # Store class
            y_a[i] = self.y_a[ID]
            y_t[i] = self.y_t[ID]

        return [X_a, X_t], {'output_a': y_a, 'output_t': y_t}


def get_model(input_length=10, n_filters=3, vocab_size=10, n_classes=9, embedding_size=5, n_modules=5, model_type='ACT',
              learning_rate=0.002):
    # inception model

    inputs = []
    for i in range(2):
        inputs.append(Input(shape=(input_length,)))

    inputs_ = []
    for i in range(2):
        if (i == 0):
            a = Embedding(vocab_size, embedding_size, input_length=input_length)(inputs[0])
            inputs_.append(Embedding(vocab_size, embedding_size, input_length=input_length)(inputs[i]))
        else:
            inputs_.append(Reshape((input_length, 1))(inputs[i]))

    filters_inputs = Concatenate(axis=2)(inputs_)

    for m in range(n_modules):
        filters = []
        for i in range(n_filters):
            filters.append(
                Conv1D(filters=32, strides=1, kernel_size=1 + i, activation='relu', padding='same')(filters_inputs))
        filters.append(MaxPooling1D(pool_size=3, strides=1, padding='same')(filters_inputs))
        filters_inputs = Concatenate(axis=2)(filters)
        # filters_inputs = Dropout(0.1)(filters_inputs)

    # pool = GlobalAveragePooling1D()(filters_inputs)
    pool = GlobalMaxPooling1D()(filters_inputs)
    # pool = Flatten()(filters_inputs)

    # pool = Dense(64, activation='relu')(pool)

    optimizer = Adam(lr=learning_rate)

    if model_type == 'BOTH':
        out_a = Dense(n_classes, activation='softmax', name='output_a')(pool)
        out_t = Dense(1, activation='linear', name='output_t')(pool)
        model = Model(inputs=inputs, outputs=[out_a, out_t])
        model.compile(optimizer=optimizer, loss={'output_a': 'categorical_crossentropy', 'output_t': 'mae'})
    else:
        if model_type == 'ACT':
            out = Dense(n_classes, activation='softmax')(pool)
            model = Model(inputs=inputs, outputs=out)
            model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])
        elif model_type == 'TIME':
            out = Dense(1, activation='linear')(pool)
            model = Model(inputs=inputs, outputs=out)
            model.compile(optimizer=optimizer, loss='mae')

    model.summary()

    return model


def fit_and_score(params):
    print(params)
    start_time = perf_counter()

    model = get_model(input_length=params['input_length'], vocab_size=params['vocab_size'],
                      n_classes=params['n_classes'], model_type=params['model_type'],
                      learning_rate=params['learning_rate'], embedding_size=params['embedding_size'],
                      n_modules=params['n_modules'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)

    if params['model_type'] == 'ACT':
        h = model.fit([X_a_train, X_t_train],
                      y_a_train, epochs=200, verbose=2,
                      validation_data=([X_a_val, X_t_val], y_a_val),
                      callbacks=[early_stopping], batch_size=2 ** params['batch_size'])
    elif (params['model_type'] == 'TIME'):
        h = model.fit([X_a_train, X_t_train],
                      y_t_train, epochs=200,
                      validation_data=([X_a_val, X_t_val], y_t_val),
                      callbacks=[early_stopping], batch_size=2 ** params['batch_size'])
    else:
        h = model.fit([X_a_train, X_t_train],
                      {'output_a': y_a_train, 'output_t': y_t_train},
                      validation_data=([X_a_val, X_t_val], {"output_a": y_a_val, "output_t": y_t_val}),
                      epochs=200, verbose=0,
                      callbacks=[early_stopping], batch_size=2 ** params['batch_size'])
    #        h = model.fit_generator(generator=train_generator, validation_data=val_generator, use_multiprocessing=True, workers=8, epochs=200, callbacks=[early_stopping], max_queue_size=10000, verbose=0)

    print("Evaluation inside function: ", model.evaluate([X_a_test, X_t_test], y_a_test))

    scores = [h.history['val_loss'][epoch] for epoch in range(len(h.history['loss']))]
    score = min(scores)
    print(score)

    global best_score, best_model, best_time, best_numparameters
    end_time = perf_counter()

    if best_score > score:
        best_score = score
        best_model = model
        best_numparameters = model.count_params()
        best_time = end_time - start_time

    return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(h.history['loss']), 'n_params': model.count_params(),
            'time': end_time - start_time}


import argparse, os
from pathlib import Path
parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--dataset", help="Raw dataset to prepare", required=True)
parser.add_argument("--train", help="Start training the neural network", action="store_true")
parser.add_argument("--test", help="Start testing next event of the neural network", action="store_true")
arguments = parser.parse_args()

if not (arguments.train or arguments.test):
    print("--train or --test (or both) are required")
    sys.exit(-3)

logfile = arguments.dataset
log_filename = Path(logfile).stem
model_type = "ACT"
output_file = os.path.join("results", log_filename + ".txt")

current_time = time.strftime("%d.%m.%y-%H.%M", time.localtime())
outfile = open(output_file, 'w')

outfile.write("Starting time: %s\n" % current_time)

from pathlib import Path
import os

directory = Path(logfile).parent
filename = Path(logfile).stem
extension = ".csv"

# We need the full dataset to get the metrics
((X_a, X_t),
 (y_a, y_t),
 vocab_size,
 max_length,
 n_classes,
 divisor,
 prefix_sizes, vocabulary, y_dict) = load_data(os.path.join(directory, filename + extension))

# Load the splits from the folder
((X_a_train, X_t_train),
 (y_a_train, y_t_train),
 _, _, _, _, prefix_sizes_train, _, _) = load_data(
    os.path.join(os.path.join(directory, "train_" + filename + extension)), max_len=max_length,
    parsed_vocabulary=vocabulary, y_dict=y_dict)

((X_a_val, X_t_val),
 (y_a_val, y_t_val),
 _, _, _, _, prefix_sizes_val, _, _) = load_data(os.path.join(os.path.join(directory, "val_" + filename + extension)),
                                                 max_len=max_length, parsed_vocabulary=vocabulary, y_dict=y_dict)

((X_a_test, X_t_test),
 (y_a_test, y_t_test),
 _, _, _, _, prefix_sizes_test, _, _) = load_data(os.path.join(os.path.join(directory, "test_" + filename + extension)),
                                                  max_len=max_length, parsed_vocabulary=vocabulary, y_dict=y_dict)


emb_size = (vocab_size + 1) // 2  # --> ceil(vocab_size/2)

# For some reason loading the datasets in parts is not good
# We calculate the number of events and then split
train_len = len(X_a_train)
val_len = train_len + len(X_a_val)
print("X_A: ", len(X_a))
print("Train len: ", train_len)
print("Val len: ", val_len)
print("Test len: ", len(X_a) - val_len)
X_a_train = X_a[:train_len]
X_t_train = X_t[:train_len]
y_a_train = y_a[:train_len]
y_t_train = y_t[:train_len]
X_a_val = X_a[train_len:val_len]
X_t_val = X_t[train_len:val_len]
y_a_val = y_a[train_len:val_len]
y_t_val = y_t[train_len:val_len]
X_a_test = X_a[val_len:]
X_t_test = X_t[val_len:]
y_a_test = y_a[val_len:]
y_t_test = y_t[val_len:]

print("SHAPES: ")
print("Shape x_a_train", X_a_train.shape)
print("Shape X_t_train: ", X_t_train.shape)
print("Shape y_a_train: ", y_a_train.shape)
print("Shape x_a_val", X_a_val.shape)
print("Shape X_t_val: ", X_t_val.shape)
print("Shape y_a_val: ", y_a_val.shape)

# normalizing times
X_t_train = X_t_train / np.max(X_t)
X_t_val = X_t_val / np.max(X_t)
X_t_test = X_t_test / np.max(X_t)
# categorical output
y_a_train = to_categorical(y_a_train, num_classes=n_classes)
y_a_val = to_categorical(y_a_val, num_classes=n_classes)
y_a_test = to_categorical(y_a_test, num_classes=n_classes)


n_iter = 20

space = {'input_length': max_length, 'vocab_size': vocab_size, 'n_classes': n_classes, 'model_type': model_type,
         'embedding_size': emb_size,
         'n_modules': hp.choice('n_modules', [1, 2, 3]),
         'batch_size': hp.choice('batch_size', [9, 10]),
         'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01))}

final_brier_scores = []
final_accuracy_scores = []
final_mae_scores = []
final_mse_scores = []

p = np.random.RandomState(seed=42)

# model selection
print('Starting model selection...')
best_score = np.inf
best_model = None
best_time = 0
best_numparameters = 0

import json
trials = Trials()
if arguments.train:
    best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=n_iter, trials=trials,
                rstate=p)
    best_params = hyperopt.space_eval(space, best)
    best_model.save_weights(os.path.join("results", log_filename + "_model.h5"))
    outfile.write("\nHyperopt trials")
    outfile.write("\ntid,loss,learning_rate,n_modules,batch_size,time,n_epochs,n_params,perf_time")
    for trial in trials.trials:
        outfile.write("\n%d,%f,%f,%d,%d,%s,%d,%d,%f" % (trial['tid'],
                                                        trial['result']['loss'],
                                                        trial['misc']['vals']['learning_rate'][0],
                                                        int(trial['misc']['vals']['n_modules'][0] + 1),
                                                        trial['misc']['vals']['batch_size'][0] + 7,
                                                        (trial['refresh_time'] - trial['book_time']).total_seconds(),
                                                        trial['result']['n_epochs'],
                                                        trial['result']['n_params'],
                                                        trial['result']['time']))
    outfile.write("\n\nBest parameters:")
    print(best_params, file=outfile)
    outfile.write("\nModel parameters: %d" % best_numparameters)
    outfile.write('\nBest Time taken: %f' % best_time)

    json_str = json.dumps(best_params)
    with open(os.path.join("results", log_filename + "_parameters.json"), "w") as f:
        f.write(json_str)

if arguments.test:
    json_params = {}
    with open(os.path.join("results", log_filename + "_parameters.json"), "r") as f:
        json_params = json.load(f)
    # TODO: for some reason we cannot save the model since loading it will
    # give random predictions.
    # When testing we have to retrain the model using the best hyperaparams saved

    fit_and_score(json_params)

    #loaded_model = tf.keras.models.load_model(os.path.join("results", log_filename + "/model"))

    # evaluate
    print('Evaluating final model...')
    preds_a = best_model.predict([X_a_test, X_t_test])
    print("Tensorflow evaluation: ")
    best_model.evaluate([X_a_test, X_t_test], y_a_test)
    """
    brier_score = np.mean(
        list(map(lambda x: brier_score_loss(y_a_test[x], preds_a[x]), [i[0] for i in enumerate(y_a_test)])))
    """


    def calculate_brier_score(y_pred, y_true):
        # From: https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))


    brier_score = calculate_brier_score(preds_a, y_a_test)

    y_a_test_max = np.argmax(y_a_test, axis=1)
    preds_a_max = np.argmax(preds_a, axis=1)

    with open(os.path.join("results", "raw_" + log_filename + ".txt"), "w") as raw_file:
        raw_file.write("prefix_length;ground_truth;predicted;prediction_probs\n")
        for X, real, pred, probs in zip(X_a_test, y_a_test_max, preds_a_max, preds_a):
            raw_file.write(str(np.count_nonzero(X)) + ";" + str(real) + ";" + str(pred) + ";" + np.array2string(probs, separator=",", max_line_width=99999) + "\n")



    outfile.write("\nBrier score: %f" % brier_score)
    final_brier_scores.append(brier_score)

    accuracy = accuracy_score(y_a_test_max, preds_a_max)
    outfile.write("\nAccuracy: %f" % accuracy)
    final_accuracy_scores.append(accuracy)

    outfile.write(np.array2string(confusion_matrix(y_a_test_max, preds_a_max), separator=", "))

    outfile.flush()

    print("\n\nFinal Brier score: ", final_brier_scores, file=outfile)
    print("Final Accuracy score: ", final_accuracy_scores, file=outfile)

    from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score

    mcc = matthews_corrcoef(y_a_test_max, preds_a_max)
    precision = precision_score(y_a_test_max, preds_a_max, average="weighted")
    recall = recall_score(y_a_test_max, preds_a_max, average="weighted")
    f1 = f1_score(y_a_test_max, preds_a_max, average="weighted")
    outfile.write("\nMCC: " + str(mcc))
    outfile.write("\nWeighted Precision: " + str(precision))
    outfile.write("\nWeighted Recall: " + str(recall))
    outfile.write("\nWeighted F1: " + str(f1))

    outfile.close()
