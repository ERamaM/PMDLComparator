import itertools
import json
import numpy as np
from numpy.random import seed
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Input, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Model
from DREAM.pydream.predictive.nap.NAP import multiclass_roc_auc_score
from DREAM.pydream.util.TimedStateSamples import TimedStateSample
import json

np.seterr(divide='ignore', invalid='ignore')

class NAPr:
    def __init__(self, tss_full_log=None, tss_train_file=None, tss_val_file=None, tss_test_file=None, options=None):
        """ Options """
        self.opts = {"seed" : 42,
                     "n_epochs" : 100,
                     "n_batch_size" : 64,
                     "dropout_rate" : 0.2,
                     "eval_size" : 0.2,
                     "activation_function" : "relu"}
        self.setSeed()

        if options is not None:
            for key in options.keys():
                self.opts[key] = options[key]

        """ Load data and setup """
        if tss_test_file is not None:
            _, _, self.Y = self.loadData(tss_full_log)
            self.X_train, self.R_train, self.Y_train = self.loadData(tss_train_file)
            self.X_val, self.R_val, self.Y_val = self.loadData(tss_val_file)
            self.X_test, self.R_test, self.Y_test = self.loadData(tss_test_file)
            print("X train shape: ", self.X_train.shape)
            print("R train shape: ", self.R_train.shape)
            print("Y train shape: ", self.Y_train.shape)
            print("X test shape: ", self.X_test.shape)
            print("R test shape: ", self.R_test.shape)
            print("Y test shape: ", self.Y_test.shape)

            self.oneHotEncoderSetup()
            self.Y_train = np.asarray(
                self.onehot_encoder.transform(self.label_encoder.transform(self.Y_train).reshape(-1, 1)))
            self.Y_test = np.asarray(
                self.onehot_encoder.transform(self.label_encoder.transform(self.Y_test).reshape(-1, 1)))
            self.Y_val = np.asarray(
                self.onehot_encoder.transform(self.label_encoder.transform(self.Y_val).reshape(-1, 1)))

            self.stdScaler = MinMaxScaler()
            self.stdScaler.fit(self.X_train)
            self.X_train = self.stdScaler.transform(self.X_train)
            self.X_val = self.stdScaler.transform(self.X_val)
            self.X_test = self.stdScaler.transform(self.X_test)

            self.stdScaler_res = MinMaxScaler()
            self.stdScaler_res.fit(self.R_train)
            self.R_train = self.stdScaler_res.transform(self.R_train)
            self.R_val = self.stdScaler_res.transform(self.R_val)
            self.R_test = self.stdScaler_res.transform(self.R_test)

            self.X_train = np.concatenate([self.X_train, self.R_train], axis=1)
            self.X_val = np.concatenate([self.X_val, self.R_val], axis=1)
            self.X_test = np.concatenate([self.X_test, self.R_test], axis=1)

            """
            self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train, test_size=self.opts["eval_size"], random_state=self.opts["seed"],
                                                              shuffle=True)
            """

            insize = self.X_train.shape[1]
            outsize = len(self.Y_train[0])

            class_inputs = Input(shape=(insize,))
            l0 = Dropout(self.opts["dropout_rate"])(class_inputs)
            bn0 = BatchNormalization()(l0)
            ae_x = Dense(300, activation=self.opts["activation_function"])(bn0)
            bn1 = BatchNormalization()(ae_x)
            l1 = Dropout(self.opts["dropout_rate"])(bn1)
            l1 = Dense(200, activation=self.opts["activation_function"], name="classifier_l1")(l1)
            bn2 = BatchNormalization()(l1)
            l2 = Dropout(self.opts["dropout_rate"])(bn2)
            l2 = Dense(100, activation=self.opts["activation_function"], name="classifier_l2")(l2)
            bn3 = BatchNormalization()(l2)
            l2 = Dropout(self.opts["dropout_rate"])(bn3)
            l3 = Dense(50, activation=self.opts["activation_function"], name="classifier_l3")(l2)
            bn4 = BatchNormalization()(l3)
            l3 = Dropout(self.opts["dropout_rate"])(bn4)
            out = Dense(outsize, activation='softmax', name="classifier")(l3)

            self.model = Model([class_inputs], [out])
            losses = {
                "classifier": "categorical_crossentropy",
            }
            self.model.compile(optimizer='adam', loss=losses, metrics=['accuracy'])

    def train(self, checkpoint_path, name, save_results=False):
        event_dict_file = str(checkpoint_path) + "/" + str(name) + "_napr_onehotdict.json"
        with open(str(event_dict_file), 'w') as outfile:
            json.dump(self.one_hot_dict, outfile)

        with open(checkpoint_path + "/" + name + "_napr_model.json", 'w') as f:
            f.write(self.model.to_json())

        ckpt_file = str(checkpoint_path) + "/" + str(name) + "_napr_weights.hdf5"
        checkpoint = ModelCheckpoint(ckpt_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        hist = self.model.fit([self.X_train], [self.Y_train], batch_size=self.opts["n_batch_size"], epochs=self.opts["n_epochs"], shuffle=True,
                         validation_data=([self.X_val], [self.Y_val]),
                         callbacks=[checkpoint])
        joblib.dump(self.stdScaler, str(checkpoint_path) + "/" + str(name) + "_napr_stdScaler.pkl")
        joblib.dump(self.stdScaler_res, str(checkpoint_path) + "/" + str(name) + "_napr_stdScaler_res.pkl")
        if save_results:
            results_file = str(checkpoint_path) + "/" + str(name) + "_napr_results.json"
            with open(str(results_file), 'w') as outfile:
                json.dump(str(hist.history), outfile)

    def oneHotEncoderSetup(self):
        """ Events to One Hot"""
        events = np.unique(self.Y)

        self.label_encoder = LabelEncoder()
        integer_encoded = self.label_encoder.fit_transform(events)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.onehot_encoder.fit(integer_encoded)

        self.one_hot_dict = {}
        for event in events:
            self.one_hot_dict[event] = list(self.onehot_encoder.transform([self.label_encoder.transform([event])])[0])

    def loadData(self, file):
        x, r, y = [], [], []
        with open(file) as json_file:
            tss = json.load(json_file)
            for sample in tss:
                if sample["nextEvent"] is not None:
                    x.append(list(itertools.chain(sample["TimedStateSample"][0], sample["TimedStateSample"][1], sample["TimedStateSample"][2])))
                    r.append(list(sample["TimedStateSample"][3]))
                    y.append(sample["nextEvent"])
        return np.array(x), np.array(r), np.array(y)

    def setSeed(self):
        seed(self.opts["seed"])
        tf.compat.v1.set_random_seed(self.opts["seed"])

    def loadModel(self, path, name):
        with open(path + "/" + name + "_napr_model.json", 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(path + "/" + name + "_napr_weights.hdf5")
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        with open(path + "/" + name + "_napr_onehotdict.json", 'r') as f:
            self.one_hot_dict = json.load(f)
        # When we load the model, we do not need the scalers in fact, only the weights
        #self.stdScaler = joblib.load(path + "/" + name + "_napr_stdScaler.pkl")
        #self.stdScaler_res = joblib.load(path + "/" + name + "_napr_stdScaler_res.pkl")

    def intToEvent(self, value):
        one_hot = list(np.eye(len(self.one_hot_dict.keys()))[value])
        for k, v in self.one_hot_dict.items():
            if str(v) == str(one_hot):
                return k

    def predict(self, tss):
        """
        Predict from a list TimedStateSamples

        :param tss: list<TimedStateSamples>
        :return: tuple (DREAM-NAP output, translated next event)
        """
        if not isinstance(tss, list) or not isinstance(tss[0], TimedStateSample) :
            raise ValueError("Input is not a list with TimedStateSample")

        preds = []
        next_events = []
        for sample in tss:
            features = [list(itertools.chain(sample.export()["TimedStateSample"][0], sample.export()["TimedStateSample"][1], sample.export()["TimedStateSample"][2]))]
            features = self.stdScaler.transform(features)
            r = [list(sample.export()["TimedStateSample"][3])]
            r = self.stdScaler_res.transform(r)
            features= np.concatenate([features, r], axis=1)

            pred = np.argmax(self.model.predict(features), axis=1)
            preds.append(pred[0])
            for p in pred:
                next_events.append(self.intToEvent(p))
        return preds, next_events

    """ Callback """
    def test(self, out_file_path, write_preds_path=None):
        logs = {}
        y_pred_one_hot = self.model.predict(self.X_test)
        y_pred = y_pred_one_hot.argmax(axis=1)
        self.Y_test_int = np.argmax(self.Y_test, axis=-1)

        def calculate_brier_score(y_pred, y_true):
            # From: https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
            return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))

        test_acc = accuracy_score(self.Y_test_int, y_pred, normalize=True)
        test_mcc = matthews_corrcoef(self.Y_test_int, y_pred)
        test_brier = calculate_brier_score(self.Y_test, y_pred_one_hot)
        print("Test acc: ", test_acc)
        test_loss, _ = self.model.evaluate(self.X_test, self.Y_test)

        precision, recall, fscore, _ = precision_recall_fscore_support(self.Y_test_int, y_pred, average='weighted',
                                                                       pos_label=None)
        auc = multiclass_roc_auc_score(self.Y_test_int, y_pred, average="weighted")

        logs['test_acc'] = test_acc
        logs["test_mcc"] = test_mcc
        logs["test_brier_score"] = test_brier
        logs['test_prec_weighted'] = precision
        logs['test_rec_weighted'] = recall
        logs['test_loss'] = test_loss
        logs['test_fscore_weighted'] = fscore
        logs['test_auc_weighted'] = auc

        precision, recall, fscore, support = precision_recall_fscore_support(self.Y_test_int, y_pred,
                                                                             average='macro', pos_label=None)
        auc = multiclass_roc_auc_score(self.Y_test_int, y_pred, average="macro")
        logs['test_prec_mean'] = precision
        logs['test_rec_mean'] = recall
        logs['test_fscore_mean'] = fscore
        logs['test_auc_mean'] = auc

        json_str = json.dumps(logs, indent=4)
        with open(out_file_path, "w") as out_file:
            out_file.write(json_str)

        if write_preds_path is not None:
            raise NotImplementedError("Writing raw results is not yet implemented.")

        return logs

    def perform_test(self, out_file_path, out_preds_path):
        logs = self.test(out_file_path, write_preds_path=None)
        return logs

