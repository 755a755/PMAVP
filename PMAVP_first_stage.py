import os, sys, re

from keras_bert import load_trained_model_from_checkpoint
import pandas as pd
import tensorflow as tf
from keras.utils import np_utils
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Input, Dropout, Embedding, Flatten, MaxPooling1D, Conv1D, SimpleRNN, LSTM, GRU, \
    Multiply, GlobalMaxPooling1D, Lambda
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc, fbeta_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import time
from keras import backend as K
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from protein_encoding import PC_6
from transformers import TFAutoModelForSequenceClassification, BertTokenizer, TFBertModel, BertModel, \
    BertForPreTraining, PretrainedConfig, \
    BertConfig

if tf.config.list_physical_devices('GPU') != []:
    visible_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([visible_devices[0]],'GPU')


# with tf.device("/gpu:0"):
def focal_loss(gamma=1., alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon() + pt_1)) \
            - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(
                1. - pt_0 + K.epsilon()))

    return focal_loss_fixed


def to_one_hot(labels, dimension):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


class myCallback(tf.keras.callbacks.Callback):
    data = []

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        loss = logs.get('loss')
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        self.data.append([epoch, acc, loss, val_acc, val_loss])

    def on_train_end(self, logs=None):
        pass

    def get_data(self):
        return self.data

    def reset_data(self, logs=None):
        self.data = []


def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    F1 = fbeta_score(labels, pred_y, beta=1)
    return acc, precision, sensitivity, specificity, MCC, F1


def mcc_metric(y_true, y_pred):


    y_pred_pos = tf.cast(y_pred[:, 1] > 0.5, tf.float32)
    y_true_pos = tf.cast(y_true[:, 1] > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true_pos * y_pred_pos)
    tn = tf.reduce_sum((1 - y_true_pos) * (1 - y_pred_pos))
    fp = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    fn = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))

    num = (tp * tn) - (fp * fn)
    den = tf.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return num / (den + K.epsilon())

def seq_to_num(line, seq_length):
    seq = np.zeros(seq_length)
    for j in range(len(line)):
        seq[j] = protein_dict[line[j]]
    return seq


def readFasta(file):
    if os.path.exists(file) == False:
        print('Error: "' + file + '" does not exist.')
        sys.exit(1)

    with open(file) as f:
        records = f.read()

    if re.search('>',
                 records) == None:  # Scan through string looking for a match to the pattern, returning a match object, or None if no match was found
        print('The input file seems not in fasta format.')
        sys.exit(1)

    records = records.split('>')[1:]
    myFasta = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
        myFasta.append([name, sequence])
    return myFasta


reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=10, verbose=1)

class MambaBlock(layers.Layer):

    def __init__(self, d_model, d_state=128, d_conv=4, expand=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout

        self.norm = layers.LayerNormalization(epsilon=1e-6)

        inner_dim = d_model * expand

        self.in_proj = layers.Dense(inner_dim * 2)

        self.dwconv = layers.Conv1D(
            filters=inner_dim,
            kernel_size=d_conv,
            padding="same",
            groups=inner_dim
        )

        self.x_proj = layers.Dense(d_state, activation="swish")
        self.state_proj = layers.Dense(inner_dim, activation="swish")

        self.out_proj = layers.Dense(d_model)

        self.drop = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        residual = inputs
        x = self.norm(inputs)

        # [B, L, 2*inner_dim]
        xz = self.in_proj(x)
        x_part, z_part = tf.split(xz, 2, axis=-1)

        z_part = tf.nn.silu(z_part)

        #mixing
        x_part = self.dwconv(x_part)
        x_part = tf.nn.silu(x_part)

        s = self.x_proj(x_part)
        s = self.state_proj(s)

        x = s * z_part

        x = self.out_proj(x)
        x = self.drop(x, training=training)

        return residual + x

import tensorflow as tf
from tensorflow.keras import layers, Model

def build_protein_classifier_mamba(seq_len=121, feature_dim=1024, alpha_out1=None, alpha_out2=None,
                                   gamma=2.0):
    inputs = layers.Input(shape=(seq_len, feature_dim), name="protein_features")

    x = layers.Dense(1024, activation='relu')(inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # x = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
    # x = layers.LayerNormalization(epsilon=1e-6)(x)

    x = MambaBlock(d_model=1024, d_state=128, d_conv=4, expand=2, dropout=0.1)(x)
    x = MambaBlock(d_model=1024, d_state=128, d_conv=4, expand=2, dropout=0.1)(x)

    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    combined = layers.Concatenate()([avg_pool, max_pool])   # [1024]

    switch_f = layers.Dense(1024, activation='relu')(combined)
    # switch_f = layers.Dropout(0.2)(switch_f)
    # switch_f = layers.Dense(512, activation='relu')(switch_f)
    # switch_f = layers.LayerNormalization(epsilon=1e-6)(switch_f)

    outputs_1 = layers.Dropout(0.1)(switch_f)
    outputs_1 = Dense(512, activation='relu')(outputs_1)
    # outputs_1 = layers.BatchNormalization()(outputs_1)
    outputs_1 = Dropout(0.1)(outputs_1)
    outputs_1 = Dense(256, activation='relu')(outputs_1)
    outputs_1 = Dropout(0.1)(outputs_1)
    outputs_1 = layers.Dense(2, activation='softmax', name="binary_output")(outputs_1)

    model = Model(inputs=inputs, outputs=outputs_1)
    

    # model.compile(loss=losses, optimizer=tf.optimizers.Adam(0.0001), metrics=["accuracy"])

    model.compile(loss=focal_loss(), optimizer=tf.optimizers.Adam(0.0001), metrics=metrics_list)  #"BinaryCrossentropy"  loss=focal_loss()
    # model.compile(loss="BinaryCrossentropy", optimizer=tf.optimizers.Adam(0.0001), metrics=metrics_list)
    return model


metrics_list = [
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    mcc_metric
]


label_pos = np.ones((2762, 1), dtype=int)
label_neg = np.zeros((10089, 1), dtype=int)
label = np.append(label_pos, label_neg)

seq_length = 121


input_data3 = np.load('/media/ProtT5.npy')
print(f"input_data3:{input_data3.shape}")

original_time = time.time()

evaluation = []
evaluation.append(['Original time: {0}'.format(original_time)])
data = []
end_time = 0
file_dir = '/media/{0},{1}'.format(time.strftime("%Y-%m-%d", time.localtime()), original_time)
os.makedirs(file_dir)


callback = myCallback()
 

mean_acc = []
mean_precision = []
mean_sensitivity = []
mean_specificity = []
mean_MCC = []
mean_AUC = []
mean_AUPR = []
mean_F1 = []
config3 = './model/model/1kmer_model/1kmer_model'
for i in range(1):
    fault_list = []
    model = build_protein_classifier_mamba()
    model.summary()
    evaluation.append((['model_trainable_weights is'], ['{0}'.format(len(model.trainable_weights))]))

    seed = 520

    train_set3, test_set3, label_train, label_test = train_test_split(
        input_data3,
        label, test_size=0.2,
        train_size=0.8,
        random_state=seed,
        shuffle=True)

    fold_time = 0
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=0o425)
    start_time = time.time()
    for train, test in kfold.split(train_set3, label_train):
        if (fold_time == 0):
            label_train = np_utils.to_categorical(label_train)

        evaluation.append(['model{0}_{1} and start time:{2}'.format(i, fold_time, start_time)])

        model.fit(train_set3[train], label_train[train],
                  validation_data=(train_set3[test], label_train[test]),
                  epochs=30,
                  batch_size=64,
                  callbacks=[callback])


        evaluation.append(['epoch', 'acc', 'loss', 'val_acc', 'val_loss'])
        data = callback.get_data()
        for y in range(len(data)):
            evaluation.append(data[y])
        callback.reset_data()
        fold_time = fold_time + 1

    model.save_weights('{2}/model{0}_{1}.h5'.format(i, fold_time, file_dir))
    end_time = time.time()
    evaluation.append(['model{0} end time:{1},and model cost :{2}'.format(i, end_time, end_time - start_time)])
    preds = model.predict(test_set3)
    preds = preds[:, 1]
    pred_y = np.rint(preds)


    acc, precision, sensitivity, specificity, MCC, F1 = calculate_performace(len(label_test), pred_y, label_test)
    fpr, tpr, _ = roc_curve(label_test, preds)
    AUC = auc(fpr, tpr)
    pre, rec, _ = precision_recall_curve(label_test, preds)
    AUPR = auc(rec, pre)


    rec = pd.DataFrame(rec)
    pre = pd.DataFrame(pre)
    rec_pre = pd.concat([rec, pre], axis=1)
    rec_pre.to_csv(f'{file_dir}/{i}_{fold_time}_rec_pre.csv', index=False, sep=',', header=['rec', 'pre'])
    fpr = pd.DataFrame(fpr)
    tpr = pd.DataFrame(tpr)
    fpr_tpr = pd.concat([fpr, tpr], axis=1)
    fpr_tpr.to_csv(f'{file_dir}/{i}_{fold_time}_fpr_tpr.csv', index=False, sep=',', header=['fpr', 'tpr'])
    label_test = pd.DataFrame(label_test)
    preds = pd.DataFrame(preds)
    pred_y = pd.DataFrame(pred_y)
    result = pd.concat([label_test, preds, pred_y], axis=1)
    result.to_csv(f'{file_dir}/{i}_{fold_time}_label_test_preds_pred_y.csv', index=False, sep=',',
                    header=['label_test', 'preds', 'pred_y'])
    
    print('model%d,acc=%f,precision=%f,sensitivity=%f,specificity=%f,MCC=%f,AUC=%f,AUPR=%f, F1=%f'
          % (i, acc, precision, sensitivity, specificity, MCC, AUC, AUPR, F1))
    evaluation.append(['acc', 'precision', 'sensitivity', 'specificity', 'MCC', 'AUC', 'AUPR', 'F1'])
    evaluation.append([str(acc), str(precision), str(sensitivity), str(specificity), str(MCC), str(AUC), str(AUPR),
                       str(F1)])
    mean_acc.append(acc)
    mean_precision.append(precision)
    mean_sensitivity.append(sensitivity)
    mean_specificity.append(specificity)
    mean_MCC.append(MCC)
    mean_AUC.append(AUC)
    mean_AUPR.append(AUPR)
    mean_F1.append(F1)

evaluation.append(
    ['mean_acc', 'mean_precision', 'mean_sensitivity', 'mean_specificity', 'mean_MCC', 'mean_AUC', 'mean_AUPR',
     'mean_F1'])
evaluation.append([np.mean(mean_acc), np.mean(mean_precision), np.mean(mean_sensitivity), np.mean(mean_specificity),
                   np.mean(mean_MCC), np.mean(mean_AUC), np.mean(mean_AUPR), np.mean(mean_F1)])

evaluation.append([np.std(mean_acc),np.std(mean_precision),np.std(mean_sensitivity),np.std(mean_specificity),
                   np.std(mean_MCC),np.std(mean_AUC),np.std(mean_AUPR),np.std(mean_F1)])

evaluation.append(["total_time:{0}".format(time.time() - original_time)])
evaluation = pd.DataFrame(evaluation)
evaluation.to_csv(r"{0}/evaluation.csv".format(file_dir), header=False, index=False)

