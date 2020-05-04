import itertools

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

idx = {}

def vectorize_log(log):
    max_len = 0
    current_idx = 1
    vectorized_log = []

    times_between_curr_and_start = []
    times_between_curr_and_prev = []
    times_since_midnight = []
    times_weekday = []

    for trace in log:
        trace_ids = []
        # 3d array
        first_time = trace[0]["time:timestamp"]
        feat_1 = []
        feat_2 = []
        feat_3 = []
        feat_4 = []

        for event_pos, event in enumerate(trace):
            if event["concept:name"] not in idx:
                idx[event["concept:name"]] = current_idx
                current_idx += 1
            trace_ids.append(idx[event["concept:name"]])

            # Calculate the time features
            # Time between current event and start of trace
            time_between_curr_and_start = (event["time:timestamp"] - first_time).total_seconds()
            # Time between current event and previous event
            if event_pos == 0:
                time_between_curr_and_prev = 0
            else:
                time_between_curr_and_prev = (event["time:timestamp"] - trace[event_pos-1]["time:timestamp"]).total_seconds()
            # The third and fourth features can be already normalized
            # They do not need features from the whole log
            # Time since midnight
            time_since_midnight = (event["time:timestamp"] - event["time:timestamp"].replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 86400
            # Weekday
            weekday = event["time:timestamp"].weekday() / 7

            feat_1.append(time_between_curr_and_start)
            feat_2.append(time_between_curr_and_prev)
            feat_3.append(time_since_midnight)
            feat_4.append(weekday)

        if "[EOC]" not in idx:
            idx["[EOC]"] = current_idx
            current_idx += 1
        if "[START]" not in idx:
            idx["[START]"] = current_idx
            current_idx += 1
        # Do not augment with EOC here. Do it in the build_inputs fn
        curr_len = len(trace_ids)
        # Calculate the maximum log length to pad sequences
        if max_len < curr_len:
            max_len = curr_len
        vectorized_log.append(trace_ids)

        times_between_curr_and_start.append(feat_1)
        times_between_curr_and_prev.append(feat_2)
        times_since_midnight.append(feat_3)
        times_weekday.append(feat_4)

    # Calculate the avg of the two first time features
    avg_time_between_curr_and_start = np.mean(list(itertools.chain(*times_between_curr_and_start)))
    avg_time_between_curr_and_prev = np.mean(list(itertools.chain(*times_between_curr_and_prev)))

    pad_sequences(vectorized_log, maxlen=max_len)
    return vectorized_log, current_idx, max_len, times_between_curr_and_prev, times_between_curr_and_start,\
           times_since_midnight, times_weekday, avg_time_between_curr_and_prev, avg_time_between_curr_and_start



# See: https://stackoverflow.com/questions/55113518/where-to-place-start-and-end-tags-in-seq2seq-translations
def build_inputs(vectorized_log, max_len, is_time=False):
    encoder_inputs = []
    decoder_inputs = []
    decoder_outputs = []
    masks = []
    for trace in vectorized_log:
        for i, event in enumerate(trace):
            encoder = trace[:i + 1]
            # If it is a time feature, do not query the vocabulary index
            if not is_time:
                decoder_input = [idx["[START]"]] + trace[i + 1:]
                decoder_output = trace[i + 1:] + [idx["[EOC]"]]
            else:
                decoder_input = [0.0] + trace[i + 1:]
                decoder_output = trace[i + 1:] + [0.0]

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
    if is_time:
        typ = "float32"
    else:
        typ = "int32"
    encoder_inputs = pad_sequences(encoder_inputs, maxlen=max_len, padding="post", dtype=typ)
    decoder_inputs = pad_sequences(decoder_inputs, maxlen=max_len, padding="post", dtype=typ)
    decoder_outputs = pad_sequences(decoder_outputs, maxlen=max_len, padding="post", dtype=typ)
    masks = pad_sequences(masks, maxlen=max_len, padding="post", dtype=typ)

    return encoder_inputs, decoder_inputs, decoder_outputs, masks


def get_batch(enc_i, dec_i, dec_o, masks, batch_size):
    n_samples = len(enc_i)
    n_batches = int(n_samples / batch_size)
    for i in range(n_batches):
        batch_enc_i = enc_i[i * batch_size : (i + 1) * batch_size]
        batch_dec_i = dec_i[i * batch_size : (i + 1) * batch_size]
        batch_dec_o = dec_o[i * batch_size : (i + 1) * batch_size]
        batch_mask = masks[i * batch_size : (i + 1) * batch_size]

        yield batch_enc_i, batch_dec_i, batch_dec_o, batch_mask, i