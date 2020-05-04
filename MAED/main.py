from pm4py.objects.log.importer.xes import factory as xes_import_factory
import argparse
from pathlib import Path
import os
import tensorflow as tf
import numpy as np
import tqdm

from dnc_v2 import DNC
from preprocessing_utilities import vectorize_log, build_inputs, get_batch
from recurrent_controller import StatelessRecurrentController
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()
file = args.dataset
file_name = Path(file).stem
parent = Path(file).parent


filename = os.path.basename(file)
log = xes_import_factory.apply(file, parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
train_log = xes_import_factory.apply(os.path.join(parent, "train_" + filename),
                                     parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
val_log = xes_import_factory.apply(os.path.join(parent, "val_" + filename),
                                   parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
test_log = xes_import_factory.apply(os.path.join(parent, "test_" + filename),
                                    parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})

vectorized_log, current_idx, max_len, _,_,_,_,_,_ = vectorize_log(log)
X_train_vectorized, _, _, train_time_between_curr_and_prev, train_time_between_curr_and_start, train_time_since_midnight, train_weekday, avg_time_between_curr_and_prev, avg_time_between_curr_and_start = vectorize_log(train_log)
X_validation_vectorized, _, _, val_time_between_curr_and_prev, val_time_between_curr_and_start, val_time_since_midnight, val_weekday, _, _ = vectorize_log(val_log)
X_test_vectorized, _, _, test_time_between_curr_and_prev, test_time_between_curr_and_start, test_time_since_midnight, test_weekday, _, _ = vectorize_log(test_log)

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
EPOCHS = 100
BATCH_SIZE = 32

n_train_samples = len(enc_train_input)
n_val_samples = len(enc_val_input)
n_train_batches = int(n_train_samples/BATCH_SIZE)
n_val_batches = int(n_val_samples / BATCH_SIZE)


with graph.as_default():
    tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v1.Session(graph=graph) as session:
        print("Current idx: ", current_idx)
        ncomputer = DNC(
            StatelessRecurrentController,
            current_idx + 1 + 4, # Sum 1 for the EOC and 4 for the time features
            current_idx+1,
            current_idx+1,
            100,
            64,
            2,
            BATCH_SIZE,
            use_mem=True,
            dual_emb=False,
            use_emb_encoder=False,
            use_emb_decoder=False,
            decoder_mode=True,
            dual_controller=True,
            write_protect=True,
            emb_size=32,
            hidden_controller_dim=100,
            use_teacher=False,
            attend_dim=0,
            sampled_loss_dim=0,
            enable_drop_out=True,
            nlayer=1,
            name='vanila'
        )

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        # Because sampled loss dim == 0
        _, prob, loss, apply_gradients = ncomputer.build_loss_function_mask(optimizer, clip_s=[-5, 5])
        session.run(tf.compat.v1.global_variables_initializer())
        for epoch in range(EPOCHS):
            #print("EPOCH ", epoch)

            losses = []
            train_pred = []
            train_real = []
            pbar = tqdm.tqdm(range(n_train_batches))
            for i in pbar:
                #print("Batch ", i, " of ", n_batches)
                batch_enc_i, batch_dec_i, batch_dec_o, batch_masks, id = next(
                    get_batch(enc_train_input, dec_train_input, dec_train_output, masks_train, BATCH_SIZE))
                # Convert ids to tensors
                batch_enc_i = tf.keras.utils.to_categorical(batch_enc_i, num_classes=current_idx+1)
                batch_dec_i = tf.keras.utils.to_categorical(batch_dec_i, num_classes=current_idx+1)
                batch_dec_o = tf.keras.utils.to_categorical(batch_dec_o, num_classes=current_idx+1)
                # Partition time features
                # Expand the dimesntions so as the dimensions are: (batch, len, 1)
                batch_enc_train_time_between_curr_and_prev = np.expand_dims(enc_train_time_between_curr_and_prev[id * BATCH_SIZE : (id+1) * BATCH_SIZE], axis=-1)
                batch_enc_train_time_between_curr_and_start = np.expand_dims(enc_train_time_between_curr_and_start[id * BATCH_SIZE : (id+1) * BATCH_SIZE], axis=-1)
                batch_enc_train_since_midnight = np.expand_dims(enc_train_since_midnight[id * BATCH_SIZE : (id+1) * BATCH_SIZE], axis=-1)
                batch_enc_train_weekday = np.expand_dims(enc_train_weekday[id * BATCH_SIZE : (id+1) * BATCH_SIZE], axis=-1)
                # Concatenate the time vectors with the activity features
                batch_enc_i = np.concatenate([
                    batch_enc_i,
                    batch_enc_train_time_between_curr_and_prev,
                    batch_enc_train_time_between_curr_and_start,
                    batch_enc_train_since_midnight,
                    batch_enc_train_weekday
                ], axis=-1)

                #print("Shape: b_e_i", np.array(batch_enc_i).shape)
                loss_value, _, out = session.run([
                    loss,
                    apply_gradients,
                    prob
                ], feed_dict={
                    ncomputer.input_encoder: batch_enc_i,
                    ncomputer.input_decoder: batch_dec_i,
                    ncomputer.target_output: batch_dec_o,
                    ncomputer.sequence_length: max_len,
                    ncomputer.decode_length: max_len,
                    ncomputer.mask: batch_masks,
                    # ncomputer.teacher_force: ncomputer.get_bool_rand_incremental(decoder_length, prob_true_max=0.5),
                    ncomputer.drop_out_keep: 0.2
                })
                predicted_next_event = np.argmax(out[:, 0, :], axis=-1)
                real_next_event = np.argmax(batch_dec_o[:, 0, :], axis=-1)
                for predicted, real in zip(predicted_next_event, real_next_event):
                    train_pred.append(predicted)
                    train_real.append(real)
                train_acc = accuracy_score(train_real, train_pred)
                losses.append(loss_value)
                pbar.set_description_str("Epoch  " + str(epoch) + "/" + str(EPOCHS) + " | Loss " + str(np.mean(losses)) + " | Train acc: " + str(train_acc))
                pbar.update()
            print("Epoch loss: ", np.mean(losses))

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
                batch_enc_val_time_between_curr_and_prev = np.expand_dims(enc_val_time_between_curr_and_prev[id * BATCH_SIZE : (id+1) * BATCH_SIZE], axis=-1)
                batch_enc_val_time_between_curr_and_start = np.expand_dims(enc_val_time_between_curr_and_start[id * BATCH_SIZE : (id+1) * BATCH_SIZE], axis=-1)
                batch_enc_val_since_midnight = np.expand_dims(enc_val_since_midnight[id * BATCH_SIZE : (id+1) * BATCH_SIZE], axis=-1)
                batch_enc_val_weekday = np.expand_dims(enc_val_weekday[id * BATCH_SIZE : (id+1) * BATCH_SIZE], axis=-1)
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
                    # ncomputer.teacher_force: ncomputer.get_bool_rand_incremental(decoder_length, prob_true_max=0.5),
                    ncomputer.drop_out_keep: 0.2
                })
                predicted_next_event = np.argmax(out[:, 0, :], axis=-1)
                real_next_event = np.argmax(batch_dec_o[:, 0, :], axis=-1)
                for pred, real in zip(predicted_next_event, real_next_event):
                    val_pred.append(pred)
                    val_real.append(real)
                val_losses.append(loss_value)
            val_acc = accuracy_score(val_pred, val_real)
            print("Validation acc: ", val_acc, " | Validation loss: ", np.mean(val_losses))

