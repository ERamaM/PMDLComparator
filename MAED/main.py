from pm4py.objects.log.importer.xes import factory as xes_import_factory
import argparse
from pathlib import Path
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import tqdm

from MAED.dnc_v2 import DNC
from MAED.recurrent_controller import StatelessRecurrentController
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description="Run the neural net")
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()
file = args.dataset
file_name = Path(file).stem
parent = Path(file).parent

idx = {}


def vectorize_log(log):
    max_len = 0
    current_idx = 1
    vectorized_log = []
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
        if "[START]" not in idx:
            idx["[START]"] = current_idx
            current_idx += 1
        # Do not augment with EOC here. Do it in the build_inputs fn
        """
        trace_ids.append(idx["[EOC]"])
        """
        curr_len = len(trace_ids)
        # Calculate the maximum log length to pad sequences
        if max_len < curr_len:
            max_len = curr_len
        vectorized_log.append(trace_ids)

    pad_sequences(vectorized_log, maxlen=max_len)
    return vectorized_log, current_idx, max_len


# See: https://stackoverflow.com/questions/55113518/where-to-place-start-and-end-tags-in-seq2seq-translations
def build_inputs(vectorized_log, max_len):
    encoder_inputs = []
    decoder_inputs = []
    decoder_outputs = []
    masks = []
    for trace in vectorized_log:
        for i, event in enumerate(trace):
            encoder = trace[:i + 1]
            decoder_input = [idx["[START]"]] + trace[i + 1:]
            decoder_output = trace[i + 1:] + [idx["[EOC]"]]
            mask = [1] * len(decoder_output)
            encoder_inputs.append(encoder)
            decoder_inputs.append(decoder_input)
            decoder_outputs.append(decoder_output)
            masks.append(mask)
            # Stop when the decoder input only has start as inputs and the decoder output
            # only has EOC as output
            if i + 1 == len(trace):
                break

    # Pad sequences
    encoder_inputs = pad_sequences(encoder_inputs, maxlen=max_len, padding="post")
    decoder_inputs = pad_sequences(decoder_inputs, maxlen=max_len, padding="post")
    decoder_outputs = pad_sequences(decoder_outputs, maxlen=max_len, padding="post")
    masks = pad_sequences(masks, maxlen=max_len, padding="post")

    return encoder_inputs, decoder_inputs, decoder_outputs, masks


def get_batch(enc_i, dec_i, dec_o, masks, batch_size):
    n_samples = len(enc_i)
    n_batches = int(n_samples / batch_size)
    for i in range(n_batches):
        batch_enc_i = enc_i[i * batch_size : (i + 1) * batch_size]
        batch_dec_i = dec_i[i * batch_size : (i + 1) * batch_size]
        batch_dec_o = dec_o[i * batch_size : (i + 1) * batch_size]
        batch_mask = masks[i * batch_size : (i + 1) * batch_size]

        yield batch_enc_i, batch_dec_i, batch_dec_o, batch_mask


filename = os.path.basename(file)
log = xes_import_factory.apply(file, parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
train_log = xes_import_factory.apply(os.path.join(parent, "train_" + filename),
                                     parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
val_log = xes_import_factory.apply(os.path.join(parent, "val_" + filename),
                                   parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
test_log = xes_import_factory.apply(os.path.join(parent, "test_" + filename),
                                    parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})

vectorized_log, current_idx, max_len = vectorize_log(log)
X_train_vectorized, _, _ = vectorize_log(train_log)
X_validation_vectorized, _, _ = vectorize_log(val_log)
X_test_vectorized, _, _ = vectorize_log(test_log)

enc_input, dec_input, dec_output, masks = build_inputs(vectorized_log, max_len)
enc_train_input, dec_train_input, dec_train_output, masks_train = build_inputs(X_train_vectorized, max_len)
enc_val_input, dec_val_input, dec_val_output, masks_val = build_inputs(X_validation_vectorized, max_len)
enc_test_input, dec_test_input, dec_test_output, masks_test = build_inputs(X_test_vectorized, max_len)

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
            current_idx+1,
            current_idx+1,
            current_idx+1,
            256,
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
            emb_size=64,
            hidden_controller_dim=256,
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
            train_acc = []
            pbar = tqdm.tqdm(range(n_train_batches))
            for i in pbar:
                #print("Batch ", i, " of ", n_batches)
                batch_enc_i, batch_dec_i, batch_dec_o, batch_masks = next(get_batch(enc_train_input, dec_train_input, dec_train_output, masks_train, BATCH_SIZE))
                # Convert ids to tensors
                batch_enc_i = tf.keras.utils.to_categorical(batch_enc_i, num_classes=current_idx+1)
                batch_dec_i = tf.keras.utils.to_categorical(batch_dec_i, num_classes=current_idx+1)
                batch_dec_o = tf.keras.utils.to_categorical(batch_dec_o, num_classes=current_idx+1)
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
                train_acc.append(accuracy_score(real_next_event, predicted_next_event))
                losses.append(loss_value)
                pbar.set_description_str("Epoch  " + str(epoch) + "/" + str(EPOCHS) + " | Loss " + str(np.mean(losses)) + " | Train acc: " + str(np.mean(train_acc)))
                pbar.update()
            print("Epoch loss: ", np.mean(losses))

            val_acc = []
            for batch in range(n_val_batches):
                # Start validation
                batch_enc_i, batch_dec_i, batch_dec_o, batch_masks = next(
                    get_batch(enc_val_input, dec_val_input, dec_val_output, masks_val, BATCH_SIZE))
                # Convert ids to tensors
                batch_enc_i = tf.keras.utils.to_categorical(batch_enc_i, num_classes=current_idx + 1)
                batch_dec_i = tf.keras.utils.to_categorical(batch_dec_i, num_classes=current_idx + 1)
                batch_dec_o = tf.keras.utils.to_categorical(batch_dec_o, num_classes=current_idx + 1)
                # print("Shape: b_e_i", np.array(batch_enc_i).shape)
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
                val_acc.append(accuracy_score(real_next_event, predicted_next_event))
            print("Validation acc: ", np.mean(val_acc))

"""
idx["[EOC]"] = 5
idx["[START]"] = 6
enc_i, dec_i, dec_o, masks = build_inputs([[1, 2, 3, 4]], 4)
print("Encoder inputs: ", enc_i)
print("Decoder input: ", dec_i)
print("Decoder output: ", dec_o)
print("Masks: ", masks)
get_batch(enc_i, dec_i, dec_o, masks, 2)
"""
