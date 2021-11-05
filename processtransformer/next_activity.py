import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics 

from processtransformer import constants
from processtransformer.data import loader
from processtransformer.models import transformer

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

parser = argparse.ArgumentParser(description="Process Transformer - Next Activity Prediction.")

parser.add_argument("--dataset", required=True, type=str, help="dataset name")

parser.add_argument("--model_dir", default="./models", type=str, help="model directory")

parser.add_argument("--result_dir", default="./results", type=str, help="results directory")

parser.add_argument("--task", type=constants.Task, 
    default=constants.Task.NEXT_ACTIVITY,  help="task name")

parser.add_argument("--epochs", default=10, type=int, help="number of total epochs")

parser.add_argument("--batch_size", default=12, type=int, help="batch size")

parser.add_argument("--learning_rate", default=0.001, type=float,
                    help="learning rate")

parser.add_argument("--gpu", default=0, type=int, 
                    help="gpu id")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if __name__ == "__main__":
    # Create directories to save the results and models
    model_path = f"{args.model_dir}/{args.dataset}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = f"{model_path}/next_activity_ckpt"

    result_path = f"{args.result_dir}/{args.dataset}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = f"{result_path}/results"

    # Load data
    data_loader = loader.LogsDataLoader(name = args.dataset)

    (train_df, test_df, val_df, x_word_dict, y_word_dict, max_case_length,
        vocab_size, num_output) = data_loader.load_data(args.task)
    
    # Prepare training examples for next activity prediction task
    train_token_x, train_token_y = data_loader.prepare_data_next_activity(train_df,
        x_word_dict, y_word_dict, max_case_length)
    val_token_x, val_token_y = data_loader.prepare_data_next_activity(val_df,
                x_word_dict, y_word_dict, max_case_length)

    # Create and train a transformer model
    transformer_model = transformer.get_next_activity_model(
        max_case_length=max_case_length, 
        vocab_size=vocab_size,
        output_dim=num_output)

    transformer_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_weights_only=True,
        monitor="val_sparse_categorical_accuracy",
        mode="max", save_best_only=True)


    transformer_model.fit(train_token_x, train_token_y, validation_data=(val_token_x, val_token_y),
        epochs=args.epochs, batch_size=args.batch_size, 
        shuffle=True, verbose=2, callbacks=[model_checkpoint_callback])

    # Evaluate over all the prefixes (k) and save the results
    k, accuracies,fscores, precisions, recalls = [],[],[],[],[]
    y_true_vec = []
    y_pred_vec = []
    y_pred_prob_vec = []
    y_true_oh_vec = []
    for i in range(max_case_length):
        test_data_subset = test_df[test_df["k"]==i]
        if len(test_data_subset) > 0:
            test_token_x, test_token_y = data_loader.prepare_data_next_activity(test_data_subset, 
                x_word_dict, y_word_dict, max_case_length)
            raw_predictions = transformer_model.predict(test_token_x)
            y_pred = np.argmax(raw_predictions, axis=1)
            #accuracy = metrics.accuracy_score(test_token_y, y_pred)
            for raw_prediction, prediction in zip(raw_predictions, y_pred):
                y_pred_vec.append(prediction)
                y_pred_prob_vec.append(tf.nn.softmax(raw_prediction, axis=-1).numpy().tolist())
            for truth in test_token_y:
                y_true_vec.append(truth)
                y_true_oh_vec.append(np.eye(num_output)[int(truth)].tolist())
            precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                test_token_y, y_pred, average="weighted")
            k.append(i)
            fscores.append(fscore)
            precisions.append(precision)
            recalls.append(recall)

    k.append(i + 1)
    def calculate_brier_score(y_pred, y_true):
        # From: https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))
    acc = metrics.accuracy_score(y_true_vec, y_pred_vec)
    fscore = metrics.f1_score(y_true_vec, y_pred_vec, average="weighted")
    precision = metrics.precision_score(y_true_vec, y_pred_vec, average="weighted")
    recall = metrics.precision_score(y_true_vec, y_pred_vec, average="weighted")
    mcc = metrics.matthews_corrcoef(y_true_vec, y_pred_vec)
    brier_score = calculate_brier_score(np.array(y_pred_prob_vec), np.array(y_true_oh_vec))

    print("Saving at: ", result_path + "_next_activity.csv")
    with open(result_path + "_next_activity.csv", "w") as result_file:
        result_file.write("Accuracy: " + str(acc) + "\n")
        result_file.write("MCC: " + str(mcc) + "\n")
        result_file.write("Precision: " + str(precision) + "\n")
        result_file.write("Recall: " + str(recall) + "\n")
        result_file.write("F1: " + str(fscore) + "\n")
        result_file.write("Brier score: " + str(brier_score) + "\n")