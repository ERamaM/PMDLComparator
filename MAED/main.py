from pm4py.objects.log.importer.xes import factory as xes_import_factory
import argparse
import itertools
from pathlib import Path
import os
import tensorflow as tf
import numpy as np
import tqdm
import sys
import numpy as np

from dnc_v2 import DNC
from preprocessing_utilities import vectorize_log, build_inputs, get_batch, calculate_brier_score
from recurrent_controller import StatelessRecurrentController
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score

tf.compat.v1.disable_eager_execution()

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

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
WORD_STORED_MEMORY = 100
HIDDEN_SIZE = 100
WORD_MEMORY_SIZE = 64
MEMORY_READ_HEADS = 2
EMBEDDING_SIZE = 32
N_LAYERS = 1
DROPOUT = 0.2
LR = 3e-4
MAX_CLIP_VALUE = 1
MIN_CLIP_VALUE = -1

parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--train", help="Start training the neural network", action="store_true")
parser.add_argument("--test", help="Start testing next event of the neural network", action="store_true")
args = parser.parse_args()
file = args.dataset
file_name = Path(file).stem
real_file_name = file_name
if file_name.find(".") != -1:
    real_file_name = file_name.split(".")[0]
real_file_name = real_file_name.lower()

# Adjust hyperparameters using the ones from the paper
# Use defaults for other logs
if real_file_name == "helpdesk":
    WORD_STORED_MEMORY = 5
    WORD_MEMORY_SIZE = 20
if real_file_name == "bpi_challenge_2012_w" or real_file_name == "bpi_challenge_2012_w_complete":
    WORD_MEMORY_SIZE = 20
    WORD_STORED_MEMORY = 20

parent = Path(file).parent

if not (args.train or args.test):
    print("--train or --test (or both) are required")
    sys.exit(3)

filename = os.path.basename(file)
log = xes_import_factory.apply(file, parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
train_log = xes_import_factory.apply(os.path.join(parent, "train_" + filename),
                                     parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
val_log = xes_import_factory.apply(os.path.join(parent, "val_" + filename),
                                   parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
test_log = xes_import_factory.apply(os.path.join(parent, "test_" + filename),
                                    parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})

vectorized_log, current_idx, max_len, _, _, _, _, _, _ = vectorize_log(log)
X_train_vectorized, _, _, train_time_between_curr_and_prev, train_time_between_curr_and_start, train_time_since_midnight, train_weekday, avg_time_between_curr_and_prev, avg_time_between_curr_and_start = vectorize_log(
    train_log)
X_validation_vectorized, _, _, val_time_between_curr_and_prev, val_time_between_curr_and_start, val_time_since_midnight, val_weekday, _, _ = vectorize_log(
    val_log)
X_test_vectorized, _, _, test_time_between_curr_and_prev, test_time_between_curr_and_start, test_time_since_midnight, test_weekday, _, _ = vectorize_log(
    test_log)

# Normalize the first and second time features
# Training
for c_p, c_s in zip(train_time_between_curr_and_prev, train_time_between_curr_and_start):
    i = 0
    for t, u in zip(c_p, c_s):
        c_p[i] = t / avg_time_between_curr_and_prev
        c_s[i] = u / avg_time_between_curr_and_start
        i += 1

# Validation
for c_p, c_s in zip(val_time_between_curr_and_prev, val_time_between_curr_and_start):
    i = 0
    for t, u in zip(c_p, c_s):
        c_p[i] = t / avg_time_between_curr_and_prev
        c_s[i] = u / avg_time_between_curr_and_start
        i += 1

# Test
for c_p, c_s in zip(test_time_between_curr_and_prev, test_time_between_curr_and_start):
    i = 0
    for t, u in zip(c_p, c_s):
        c_p[i] = t / avg_time_between_curr_and_prev
        c_s[i] = u / avg_time_between_curr_and_start
        i += 1

enc_input, dec_input, dec_output, masks = build_inputs(vectorized_log, max_len)
enc_train_input, dec_train_input, dec_train_output, masks_train = build_inputs(X_train_vectorized, max_len)
enc_val_input, dec_val_input, dec_val_output, masks_val = build_inputs(X_validation_vectorized, max_len)
enc_test_input, dec_test_input, dec_test_output, masks_test = build_inputs(X_test_vectorized, max_len)

# Build the inputs for the time features
enc_train_time_between_curr_and_prev, _, _, _ = build_inputs(train_time_between_curr_and_prev, max_len, True)
enc_train_time_between_curr_and_start, _, _, _ = build_inputs(train_time_between_curr_and_start, max_len, True)
enc_train_since_midnight, _, _, _ = build_inputs(train_time_since_midnight, max_len, True)
enc_train_weekday, _, _, _ = build_inputs(train_weekday, max_len, True)

enc_val_time_between_curr_and_prev, _, _, _ = build_inputs(val_time_between_curr_and_prev, max_len, True)
enc_val_time_between_curr_and_start, _, _, _ = build_inputs(val_time_between_curr_and_start, max_len, True)
enc_val_since_midnight, _, _, _ = build_inputs(val_time_since_midnight, max_len, True)
enc_val_weekday, _, _, _ = build_inputs(val_weekday, max_len, True)

enc_test_time_between_curr_and_prev, _, _, _ = build_inputs(test_time_between_curr_and_prev, max_len, True)
enc_test_time_between_curr_and_start, _, _, _ = build_inputs(test_time_between_curr_and_start, max_len, True)
enc_test_since_midnight, _, _, _ = build_inputs(test_time_since_midnight, max_len, True)
enc_test_weekday, _, _, _ = build_inputs(test_weekday, max_len, True)

graph = tf.Graph()

# Hyperparameters

n_train_samples = len(enc_train_input)
n_val_samples = len(enc_val_input)
n_test_samples = len(enc_test_input)
n_train_batches = int(n_train_samples / BATCH_SIZE)
n_val_batches = int(n_val_samples / BATCH_SIZE)
n_test_batches = int(n_test_samples / BATCH_SIZE)

with graph.as_default():
    tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v1.Session(graph=graph) as session:
        print("Current idx: ", current_idx)
        print("Len train input: ", len(enc_train_input))

        # Convert ids to tensors
        batch_enc_i_train = tf.keras.utils.to_categorical(enc_train_input, num_classes=current_idx + 1)
        batch_dec_i_train = tf.keras.utils.to_categorical(dec_train_input, num_classes=current_idx + 1)
        batch_dec_o_train = tf.keras.utils.to_categorical(dec_train_output, num_classes=current_idx + 1)
        # Partition time features
        # Expand the dimesntions so as the dimensions are: (batch, len, 1)
        batch_enc_train_time_between_curr_and_prev = np.expand_dims(
            enc_train_time_between_curr_and_prev, axis=-1)
        batch_enc_train_time_between_curr_and_start = np.expand_dims(
            enc_train_time_between_curr_and_start, axis=-1)
        batch_enc_train_since_midnight = np.expand_dims(
            enc_train_since_midnight, axis=-1)
        batch_enc_train_weekday = np.expand_dims(enc_train_weekday,
                                                 axis=-1)
        # Concatenate the time vectors with the activity features
        batch_enc_i_train = np.concatenate([
            batch_enc_i_train,
            batch_enc_train_time_between_curr_and_prev,
            batch_enc_train_time_between_curr_and_start,
            batch_enc_train_since_midnight,
            batch_enc_train_weekday
        ], axis=-1)

        # VALIDATIO
        # Convert ids to tensors
        batch_enc_i_val = tf.keras.utils.to_categorical(enc_val_input, num_classes=current_idx + 1)
        batch_dec_i_val = tf.keras.utils.to_categorical(dec_val_input, num_classes=current_idx + 1)
        batch_dec_o_val = tf.keras.utils.to_categorical(dec_val_output, num_classes=current_idx + 1)
        # Partition time features
        # Expand the dimesntions so as the dimensions are: (batch, len, 1)
        batch_enc_val_time_between_curr_and_prev = np.expand_dims(
            enc_val_time_between_curr_and_prev, axis=-1)
        batch_enc_val_time_between_curr_and_start = np.expand_dims(
            enc_val_time_between_curr_and_start, axis=-1)
        batch_enc_val_since_midnight = np.expand_dims(
            enc_val_since_midnight, axis=-1)
        batch_enc_val_weekday = np.expand_dims(enc_val_weekday,
                                                 axis=-1)
        # Concatenate the time vectors with the activity features
        batch_enc_i_val = np.concatenate([
            batch_enc_i_val,
            batch_enc_val_time_between_curr_and_prev,
            batch_enc_val_time_between_curr_and_start,
            batch_enc_val_since_midnight,
            batch_enc_val_weekday
        ], axis=-1)

        train_dataset = tf.compat.v1.data.Dataset.from_tensor_slices((batch_enc_i_train, batch_dec_i_train, batch_dec_o_train, masks_train)).batch(BATCH_SIZE)
        val_dataset = tf.compat.v1.data.Dataset.from_tensor_slices((batch_enc_i_val, batch_dec_i_val, batch_dec_o_val, masks_val)).batch(BATCH_SIZE)
        iterator = tf.compat.v1.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        training_op = iterator.make_initializer(train_dataset)
        val_op = iterator.make_initializer(val_dataset)
        session.run(training_op)
        i_e, i_d, t_o, m = iterator.get_next()


        ncomputer = DNC(
            i_e, i_d, t_o, m,
            StatelessRecurrentController,
            current_idx + 1 + 4,  # Sum 1 for the EOC and 4 for the time features
            current_idx + 1,
            current_idx + 1,
            WORD_STORED_MEMORY,
            WORD_MEMORY_SIZE,
            MEMORY_READ_HEADS,
            BATCH_SIZE,
            use_mem=True,
            dual_emb=False,
            use_emb_encoder=False,
            use_emb_decoder=False,
            decoder_mode=True,
            dual_controller=True,
            write_protect=True,
            emb_size=EMBEDDING_SIZE,
            hidden_controller_dim=HIDDEN_SIZE,
            use_teacher=False,
            attend_dim=0,
            sampled_loss_dim=0,
            enable_drop_out=True,
            nlayer=N_LAYERS,
            name='vanila'
        )

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LR)
        # Because sampled loss dim == 0
        _, prob, loss, apply_gradients = ncomputer.build_loss_function_mask(optimizer, clip_s=[MIN_CLIP_VALUE, MAX_CLIP_VALUE])

        session.run(tf.compat.v1.global_variables_initializer())

        if args.train:
            max_val_acc = 0
            train_pred_values = []
            train_real_values = []

            session.run(training_op)
            for epoch in range(EPOCHS):
                losses = []
                train_pred = []
                train_real = []
                # print("Shape: b_e_i", np.array(batch_enc_i).shape)

                print("Run")
                progress_bar = tqdm.tqdm(range(n_train_batches))
                for b in progress_bar:
                    loss_value, _, out, _ = session.run([
                        loss,
                        apply_gradients,
                        prob, training_op
                    ], feed_dict = {
                        ncomputer.sequence_length: max_len,
                        ncomputer.decode_length: max_len,
                        ncomputer.drop_out_keep: DROPOUT
                    }
                    )
                    predicted_next_event = np.argmax(out[:, 0, :], axis=-1)
                    #print("Predicted_next: ", predicted_next_event)
                    # Run the real tensors in a session to retrieve its numpy value
                    # From the sequences select the first chars
                    real_next_event = np.argmax(session.run(t_o), axis=-1)[:, 0]
                    #print("Real next: ", real_next_event)
                    for p, r in zip(predicted_next_event, real_next_event):
                        train_pred_values.append(p)
                        train_real_values.append(r)
                    losses.append(loss_value)

                    train_acc = accuracy_score(train_real_values, train_pred_values)
                    progress_bar.set_description_str("Epoch  " + str(epoch) + "/" + str(EPOCHS) + " | Loss " + str(
                        np.mean(losses)) + " | Train acc: " + str(train_acc))

                    #print("Train acc: ", train_acc)
                    #print("Epoch loss: ", np.mean(losses))

                val_losses =[]
                val_pred = []
                val_real = []
                session.run(val_op)
                for i in range(n_val_batches):
                    val_loss_value, out, _, = session.run([
                        loss,
                        prob, val_op
                    ], feed_dict = {
                        ncomputer.sequence_length: max_len,
                        ncomputer.decode_length: max_len,
                        ncomputer.drop_out_keep: DROPOUT
                    }
                    )

                    predicted_next_event = np.argmax(out[:, 0, :], axis=-1)
                    real_next_event = np.argmax(session.run(t_o), axis=-1)[:, 0]
                    for predicted, real in zip(predicted_next_event, real_next_event):
                        val_pred.append(predicted)
                        val_real.append(real)
                    val_acc = accuracy_score(val_real, val_pred)
                    val_losses.append(val_loss_value)

                val_acc = accuracy_score(val_real, val_pred)
                print("Val loss: ", np.mean(val_losses))
                print("Val acc: ", val_acc)

                """
                pbar.set_description_str("Epoch  " + str(epoch) + "/" + str(EPOCHS) + " | Loss " + str(
                    np.mean(losses)) + " | Train acc: " + str(train_acc))
                pbar.update()
                """

"""
                val_pred = []
                val_real = []
                val_losses = []
                for batch in range(n_val_batches):
                    # Start validation
                    batch_enc_i, batch_dec_i, batch_dec_o, batch_masks, id = next(
                        get_batch(enc_val_input, dec_val_input, dec_val_output, masks_val, BATCH_SIZE))
                    # Convert ids to tensors
                    batch_enc_i = tf.keras.utils.to_categorical(batch_enc_i, num_classes=current_idx + 1)
                    batch_dec_i = tf.keras.utils.to_categorical(batch_dec_i, num_classes=current_idx + 1)
                    batch_dec_o = tf.keras.utils.to_categorical(batch_dec_o, num_classes=current_idx + 1)
                    # print("Shape: b_e_i", np.array(batch_enc_i).shape)
                    batch_enc_val_time_between_curr_and_prev = np.expand_dims(
                        enc_val_time_between_curr_and_prev[id * BATCH_SIZE: (id + 1) * BATCH_SIZE], axis=-1)
                    batch_enc_val_time_between_curr_and_start = np.expand_dims(
                        enc_val_time_between_curr_and_start[id * BATCH_SIZE: (id + 1) * BATCH_SIZE], axis=-1)
                    batch_enc_val_since_midnight = np.expand_dims(
                        enc_val_since_midnight[id * BATCH_SIZE: (id + 1) * BATCH_SIZE], axis=-1)
                    batch_enc_val_weekday = np.expand_dims(enc_val_weekday[id * BATCH_SIZE: (id + 1) * BATCH_SIZE],
                                                           axis=-1)
                    # Concatenate the time vectors with the activity features
                    batch_enc_i = np.concatenate([
                        batch_enc_i,
                        batch_enc_val_time_between_curr_and_prev,
                        batch_enc_val_time_between_curr_and_start,
                        batch_enc_val_since_midnight,
                        batch_enc_val_weekday
                    ], axis=-1)
                    loss_value, out = session.run([
                        loss,
                        prob
                    ], feed_dict={
                        ncomputer.input_encoder: batch_enc_i,
                        ncomputer.input_decoder: batch_dec_i,
                        ncomputer.target_output: batch_dec_o,
                        ncomputer.sequence_length: max_len,
                        ncomputer.decode_length: max_len,
                        ncomputer.mask: batch_masks,
                        ncomputer.teacher_force: ncomputer.get_bool_rand_incremental(max_len, prob_true_max=0),
                        ncomputer.drop_out_keep: DROPOUT
                    })
                    predicted_next_event = np.argmax(out[:, 0, :], axis=-1)
                    real_next_event = np.argmax(batch_dec_o[:, 0, :], axis=-1)
                    for pred, real in zip(predicted_next_event, real_next_event):
                        val_pred.append(pred)
                        val_real.append(real)
                    val_losses.append(loss_value)
                val_acc = accuracy_score(val_pred, val_real)
                mean_val_loss = np.mean(val_losses)
                print("Validation acc: ", val_acc, " | Validation loss: ", mean_val_loss)
                # Save model with best validation loss
                if val_acc > max_val_acc:
                    print("Val acc improved from ", max_val_acc, " to ", val_acc, ". Saving model.")
                    max_val_acc = val_acc
                    ncomputer.save(session, "models/", file_name + ".h5")

        if args.test:
            # Testing
            print("Testing...")
            ncomputer.restore(session, "models/", file_name + ".h5")
            test_pbar = tqdm.tqdm(range(n_test_batches))
            test_pred = []
            test_real = []
            test_losses = []
            test_probs = []
            test_prefix_length = []
            test_real_onehot = []
            for batch in test_pbar:
                # Start testidation
                batch_enc_i, batch_dec_i, batch_dec_o, batch_masks, id = next(
                    get_batch(enc_test_input, dec_test_input, dec_test_output, masks_test, BATCH_SIZE))
                batch_prev_enc_i = batch_enc_i
                # Convert ids to tensors
                batch_enc_i = tf.keras.utils.to_categorical(batch_enc_i, num_classes=current_idx + 1)
                batch_dec_i = tf.keras.utils.to_categorical(batch_dec_i, num_classes=current_idx + 1)
                batch_dec_o = tf.keras.utils.to_categorical(batch_dec_o, num_classes=current_idx + 1)
                # print("Shape: b_e_i", np.array(batch_enc_i).shape)
                batch_enc_test_time_between_curr_and_prev = np.expand_dims(
                    enc_test_time_between_curr_and_prev[id * BATCH_SIZE: (id + 1) * BATCH_SIZE], axis=-1)
                batch_enc_test_time_between_curr_and_start = np.expand_dims(
                    enc_test_time_between_curr_and_start[id * BATCH_SIZE: (id + 1) * BATCH_SIZE], axis=-1)
                batch_enc_test_since_midnight = np.expand_dims(
                    enc_test_since_midnight[id * BATCH_SIZE: (id + 1) * BATCH_SIZE], axis=-1)
                batch_enc_test_weekday = np.expand_dims(enc_test_weekday[id * BATCH_SIZE: (id + 1) * BATCH_SIZE],
                                                        axis=-1)
                # Concatenate the time vectors with the activity features
                batch_enc_i = np.concatenate([
                    batch_enc_i,
                    batch_enc_test_time_between_curr_and_prev,
                    batch_enc_test_time_between_curr_and_start,
                    batch_enc_test_since_midnight,
                    batch_enc_test_weekday
                ], axis=-1)
                loss_test, out = session.run([
                    loss,
                    prob
                ], feed_dict={
                    ncomputer.input_encoder: batch_enc_i,
                    ncomputer.input_decoder: batch_dec_i,
                    ncomputer.target_output: batch_dec_o,
                    ncomputer.sequence_length: max_len,
                    ncomputer.decode_length: max_len,
                    ncomputer.mask: batch_masks,
                    ncomputer.teacher_force: ncomputer.get_bool_rand_incremental(max_len, prob_true_max=0),
                    ncomputer.drop_out_keep: DROPOUT
                })
                predicted_next_event = np.argmax(out[:, 0, :], axis=-1)
                real_next_event = np.argmax(batch_dec_o[:, 0, :], axis=-1)

                # The test decoder already predicts the full suffix.
                # However, instead of predicting zeros, it predicts a series of [EOC] until the end of the trace
                batch_real_suffixes = np.argmax(batch_dec_o, axis=-1)
                batch_predicted_suffixes = np.argmax(out, axis=-1)
                #for r, p in zip(real_suffixes, predicted_suffixes):
                #    print("Real: ", r, " Predicted: ", p)
                # TODO: calculate damerau-levenshtein

                for pred, real, probs, e_i, real_onehot in zip(predicted_next_event, real_next_event, out[:, 0, :], batch_prev_enc_i, batch_dec_o[:, 0, :]):
                    test_pred.append(pred)
                    test_real.append(real)
                    test_probs.append(probs)
                    test_prefix_length.append(np.count_nonzero(e_i))
                    test_real_onehot.append(real_onehot)
                test_losses.append(loss_test)
            with open(os.path.join("results", "raw_" + real_file_name + ".txt"), "w") as raw_file:
                raw_file.write("prefix_length;ground_truth;predicted;prediction_probs\n")
                for pred, real, probs, length in zip(test_pred, test_real, test_probs, test_prefix_length):
                    raw_file.write(str(pred) + ";" + str(real) + ";" + str(np.array2string(probs, separator=",", max_line_width=99999) + ";" + str(length)) + "\n")
            with open(os.path.join("results", real_file_name + ".txt"), "w") as file:
                test_acc = accuracy_score(test_pred, test_real)
                test_loss = np.mean(test_losses)
                test_brier_score = calculate_brier_score(np.array(test_real_onehot), np.array(test_probs))
                test_mcc = matthews_corrcoef(test_pred, test_real)
                file.write("Accuracy: " + str(test_acc) + "\n")
                file.write("Loss: " + str(test_loss) + "\n")
                file.write("Brier score: " + str(test_brier_score) + "\n")
                file.write("MCC: " + str(test_mcc) + "\n")

"""
