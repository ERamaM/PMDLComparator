import os
# Disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from pm4py.objects.log.importer.xes import factory as xes_import_factory
import numpy as np
import os
import math
import itertools
import random
from similarity.normalized_levenshtein import NormalizedLevenshtein
from tensorflow.python.keras.backend import set_session

tf.compat.v1.set_random_seed(42)
random.seed(42)
np.random.seed(42)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config=config)
set_session(sess)

import argparse
from pathlib import Path
parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()
file = args.dataset
file_name = Path(file).stem

model_file_name = file_name + ".h5"

log = xes_import_factory.apply(file,
                               parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"}
                               )

vectorized_log = []
idx = {}

do_damerau = False
seq_length = 20
current_idx = 0
BATCH_SIZE = 20
BUFFER_SIZE = 10000
embedding_dim = 32
rnn_units = 32
dropout = 0.2

maxGradNorm = 5
lrDecay = 0.75
lr = 1.0

# Convert the log to ids
# This conversion creates a list of lists of events, that is, a list of traces.
for trace in log:
    trace_ids = []
    for event in trace:
        if event["concept:name"] not in idx:
            idx[event["concept:name"]] = current_idx
            current_idx += 1
        trace_ids.append(idx[event["concept:name"]])
    if "[EOC]" not in idx:
        idx["[EOC]"] = current_idx
        current_idx += 1
    trace_ids.append(idx["[EOC]"])
    vectorized_log.append(trace_ids)


# Get x and y (x shuffled one)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def split(X):
    """
    Divide el conjunto de datos SIEMPRE de la misma forma
    :param X:
    :param y:
    :return: Devuelve: x_train, y_train, x_validation, y_validation, x_test, y_test
    """
    length = len(X)
    train_length = round(length * 0.64)
    val_length = round(length * 0.8)
    X_train = X[0:train_length]
    X_val = X[train_length:val_length]
    X_test = X[val_length:]

    return X_train, X_val, X_test


def to_dataset(vectorized_log):
    vectorized_log = np.array(list(itertools.chain(*vectorized_log)))
    # List of ids
    char_dataset = tf.data.Dataset.from_tensor_slices(vectorized_log)
    # Compact them on a list of traces of length seq_length+1
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    """
    for i in dataset.take(5):
        print(i)
        """
    return dataset

# Note that this split does not split in events but in traces
X_train, X_validation, X_test = split(vectorized_log)

print("Full log events: ", sum([len(x) for x in vectorized_log]))
print("Train events: ", sum([len(x) for x in X_train]))
print("Validation events: ", sum([len(x) for x in X_validation]))
print("Test events: ", sum([len(x) for x in X_test]))

train_dataset = to_dataset(X_train)
val_dataset = to_dataset(X_validation)
test_dataset = to_dataset(X_test)

vocab_size = current_idx


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform',
                             dropout=dropout),
        tf.keras.layers.Dense(vocab_size, kernel_initializer="random_uniform", bias_initializer="random_uniform"),
    ])
    return model


model = build_model(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

model.summary()


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


# optimizer = tf.keras.optimizers.SGD(lr=lr, decay=lrDecay, clipnorm=maxGradNorm, momentum=0.9, nesterov=True)
optimizer = tf.keras.optimizers.Adam(0.001)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    model_file_name,
    monitor="val_loss",
    save_weights_only=True, save_best_only=True, verbose=1)

model.compile(
    loss=loss, optimizer=optimizer,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)

history = model.fit(train_dataset, epochs=100, callbacks=[checkpoint_callback], validation_data=val_dataset)

# Test accuracy
print("Results for: ", file_name)
model.load_weights(model_file_name)
model.evaluate(test_dataset)

# Test damerau levenshtein
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(model_file_name)
model.build(tf.TensorShape([1, None]))

def generate_text(model, start_trace):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 150

    # Converting our start string to numbers (vectorizing)

    input_eval = start_trace
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    # temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        # predictions = predictions / temperature
        # predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        predicted_id = np.argmax(predictions.numpy()[0])
        # print("Predictions: ", predictions)
        #print("ID: ", predicted_id)

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(predicted_id)
        if predicted_id == idx["[EOC]"]:
            return (start_trace + text_generated)

    return (start_trace + text_generated)

def damerau_levenshtein(X_pred, X_real):
    first_printable_chr = 33
    X_pred_arr_str = []
    for x in X_pred:
        if x is not 0:
            X_pred_arr_str.append(chr(int(x) + first_printable_chr))
    X_pred_str = "".join(X_pred_arr_str)
    X_real_arr_str = []
    for x in X_real:
        if x is not 0:
            X_real_arr_str.append(chr(int(x) + first_printable_chr))
    X_real_str = "".join(X_real_arr_str)

    print("Predicted str: ", X_pred_str)
    print("Real str: ", X_real_str)
    norm = NormalizedLevenshtein()
    return norm.similarity(X_pred_str, X_real_str)

# Calculate the number of steps for the progress bar
# TODO: this may be slow
if do_damerau:
    total_steps = 0
    for test_trace in X_test:
        total_steps += len(test_trace) - 1

    progress_bar = tf.keras.utils.Progbar(total_steps)
    print("Calculating mean Damerau similarity. Please wait.")

    damerau_distances = []
    curr_step = 0
    for test_trace in X_test:
        # Iterating over the end of the array would end with suffixes with size 1
        for i in range(len(test_trace)-1):
            prefix = test_trace[0:i+1]
            suffix = test_trace[i+1:]
            if not suffix:
                continue
            prediction = generate_text(model, prefix)
            distance = damerau_levenshtein(prediction, suffix)
            damerau_distances.append(distance)
            curr_step += 1
            progress_bar.update(curr_step)

            # print("Ground truth: ", test_trace)
            # print("Prefix: ", prefix)
            # print("Suffix: ", suffix)
            # print("Prediction: ", prediction)

    print("Mean damerau: ", np.mean(damerau_distances))
