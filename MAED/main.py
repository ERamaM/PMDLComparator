from pm4py.objects.log.importer.xes import factory as xes_import_factory
import argparse
from pathlib import Path
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np

from MAED.dnc_v2 import DNC
from MAED.recurrent_controller import StatelessRecurrentController

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

graph = tf.Graph()

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
n_samples = len(enc_input)
n_batches = int(n_samples/BATCH_SIZE)


with graph.as_default():
    tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v1.Session(graph=graph) as session:
        print("Current idx: ", current_idx)
        ncomputer = DNC(
            StatelessRecurrentController,
            current_idx+1,
            current_idx+1,
            current_idx+1,
            32,
            32,
            1,
            BATCH_SIZE,
            use_mem=True,
            dual_emb=False,
            use_emb_encoder=False,
            use_emb_decoder=False,
            decoder_mode=True,
            dual_controller=True,
            write_protect=True,
            emb_size=32,
            hidden_controller_dim=32,
            use_teacher=False,
            attend_dim=0,
            sampled_loss_dim=0,
            enable_drop_out=True,
            nlayer=2,
            name='vanila'
        )

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=3e-4)
        # Because sampled loss dim == 0
        _, prob, loss, apply_gradients = ncomputer.build_loss_function_mask(optimizer, clip_s=[-5, 5])
        session.run(tf.compat.v1.global_variables_initializer())
        for epoch in range(EPOCHS):
            print("EPOCH ", epoch)
            losses = []
            for i in range(n_batches):
                print("Batch ", i, " of ", n_batches)
                batch_enc_i, batch_dec_i, batch_dec_o, batch_masks = next(get_batch(enc_input, dec_input, dec_output, masks, BATCH_SIZE))
                # print("Batch enc i: ", batch_enc_i)
                #print("Batch dec i: ", batch_dec_i)
                #print("Batch dec o: ", batch_dec_o)
                #print("Batch_masks: ", batch_masks)
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
                print("Minibatch loss: ", loss_value)
                losses.append(loss_value)
            print("EPOCH LOSS: ", np.mean(losses))

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
