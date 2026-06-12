import os, sys, re
# import keras.losses
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Input, Dropout, Embedding, Flatten, MaxPooling1D, Conv1D, SimpleRNN, LSTM, GRU, \
    Multiply, GlobalMaxPooling1D
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc, fbeta_score, recall_score, precision_score, \
    f1_score, accuracy_score
import time
from keras import backend as K
from protein_encoding import PC_6
from transformers import TFAutoModelForSequenceClassification, BertTokenizer, TFBertModel,PretrainedConfig
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras import layers, Model
from keras.utils import np_utils


if tf.config.list_physical_devices('GPU') != []:
    visible_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([visible_devices[0]],'GPU')


def categorical_focal_loss(alpha=None, gamma=2.0, label_smoothing=0.0):
    """
    多分类 Focal Loss（适用于 softmax + one-hot 标签）

    参数：
        alpha: 
            - None：所有类别权重相同
            - list / np.array：每个类别一个权重，例如长度=7或9
        gamma: 聚焦参数，常用 2.0
        label_smoothing: 标签平滑，可设 0 或很小值如 0.01
    """
    if alpha is not None:
        alpha = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        if label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

        # 防止 log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # softmax 多分类交叉熵的逐类形式
        ce = -y_true * tf.math.log(y_pred)

        # focal 因子
        focal_weight = tf.pow(1.0 - y_pred, gamma)

        # alpha 类别权重
        if alpha is not None:
            alpha_factor = y_true * alpha
            alpha_factor = tf.reduce_sum(alpha_factor, axis=-1, keepdims=True)
            loss = alpha_factor * focal_weight * ce
        else:
            loss = focal_weight * ce

        # 对类别维度求和
        loss = tf.reduce_sum(loss, axis=-1)
        return loss

    return loss_fn


def sparse_categorical_focal_loss(alpha=None, gamma=2.0):
    """
    多分类 Focal Loss（适用于 softmax + 整数标签）
    y_true: [batch,]
    y_pred: [batch, num_classes]
    """
    if alpha is not None:
        alpha = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        batch_idx = tf.range(tf.shape(y_pred)[0], dtype=tf.int32)
        indices = tf.stack([batch_idx, y_true], axis=1)

        p_t = tf.gather_nd(y_pred, indices)
        ce = -tf.math.log(p_t)
        focal_weight = tf.pow(1.0 - p_t, gamma)

        if alpha is not None:
            alpha_t = tf.gather(alpha, y_true)
            loss = alpha_t * focal_weight * ce
        else:
            loss = focal_weight * ce

        return loss

    return loss_fn


def compute_alpha_from_sparse_labels(y, num_classes, normalize=True):
    """
    根据一维整数标签计算每个类别的 alpha
    y: shape (N,) ，元素取值范围为 [0, num_classes-1]
    num_classes: 类别数
    """
    y = np.asarray(y).astype(int).reshape(-1)

    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)   # 防止某类为0

    alpha = 1.0 / counts               # 频数越少，权重越大

    if normalize:
        alpha = alpha / np.sum(alpha) * num_classes

    return alpha

def focal_loss(alpha,gamma=2.):  #我自己写的loss方程，输入的y_true要是one-hot形式，alpha要自己给每一个类设一个比例
    def focal_loss_fixed(y_true, y_pred):
        # y_true = tf.one_hot(y_true,depth=dim)# 因为我改了损失函数为focal loss所以我要把标签变成one-hot形式，而且SKF.split不能将one-hot的标签进行划分，所以先split再one-hot
        #  y_true = tf.reshape(y_true,[dim,dim])
        log_pred = K.log(y_pred)
        log_pred = tf.cast(log_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        mul = y_true * log_pred
        log = alpha * K.pow(1.0 - y_pred, gamma) * tf.cast(mul, dtype=tf.float32)
        sum = K.sum(log,axis=1)
        loss = -K.mean(sum)
        # scs=keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        # return -K.mean(alpha * K.pow(1. - y_pred, gamma) * y_true* log_pred)
        #return -K.sum(y_true * log_pred)
        # return (alpha*K.pow(1.0-y_pred,gamma)*K.mean(keras.losses.sparse_categorical_crossentropy(y_true,y_pred)))
        return  loss
    return focal_loss_fixed

def focal_loss_first(gamma=1., alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon() + pt_1)) \
            - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(
                1. - pt_0 + K.epsilon()))  # K.epsilon()函数是其中之一，它返回一个非常小的正数，通常是机器精度的一部分，用于避免数值计算中的不稳定性

    return focal_loss_fixed


def to_one_hot(labels, dimension):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


class myCallback(tf.keras.callbacks.Callback):
    data = []

    def on_epoch_end(self, epoch, logs=None):
        acc_1 = logs.get('outputs_1_accuracy')
        acc_2 = logs.get('outputs_2_accuracy')
        loss = logs.get('loss')
        val_acc_1 = logs.get('val_outputs_1_accuracy')
        val_acc_2 = logs.get('val_outputs_2_accuracy')
        val_loss = logs.get('val_loss')
        self.data.append([epoch, acc_1, acc_2, loss, val_acc_1, val_acc_2, val_loss])

    def on_train_end(self, logs=None):
        pass

    def get_data(self):
        return self.data

    def reset_data(self, logs=None):
        self.data = []


def calculate_performace(label, pred_y):
    for i in range(len(pred_y)):
        max_value = max(pred_y[i])
        for j in range(len(pred_y[i])):
            if max_value == pred_y[i][j]:
                pred_y[i][j] = 1
            else:
                pred_y[i][j] = 0
    MacroP = precision_score(label, pred_y, average='macro')
    MacroR = recall_score(label, pred_y, average='macro')
    MacroF = f1_score(label, pred_y, average='macro')
    Accuracy = accuracy_score(label, pred_y)
    return Accuracy, MacroP, MacroR, MacroF


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


# re.sub将array[1:].upper()的字符串里“非ARNDCQEGHILKMFPSTWYV-”的统统换成“-”符号



class MambaBlock(layers.Layer):
    """
    一个简化的 Keras Mamba 风格模块：
    - 输入投影
    - 门控分支
    - 局部卷积（模拟 token mixing）
    - 状态混合投影
    - 残差连接

    这不是严格复现论文 selective scan 的完整版，
    但在 Keras 中是一个很实用的 Mamba-like 替代结构。
    """
    def __init__(self, d_model, d_state=128, d_conv=4, expand=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout

        self.norm = layers.LayerNormalization(epsilon=1e-6)

        inner_dim = d_model * expand

        # 输入投影：分成两支，一支做特征变换，一支做门控
        self.in_proj = layers.Dense(inner_dim * 2)

        # 深度可分离一维卷积，模拟 Mamba 中的局部 mixing
        self.dwconv = layers.Conv1D(
            filters=inner_dim,        # 必须等于输入通道数
            kernel_size=d_conv,
            padding="same",
            groups=inner_dim          # 关键：实现 depthwise
        )

        # 状态变换（近似 SSM mixing）
        self.x_proj = layers.Dense(d_state, activation="swish")
        self.state_proj = layers.Dense(inner_dim, activation="swish")

        # 输出投影
        self.out_proj = layers.Dense(d_model)

        self.drop = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        residual = inputs
        x = self.norm(inputs)

        # [B, L, 2*inner_dim]
        xz = self.in_proj(x)
        x_part, z_part = tf.split(xz, 2, axis=-1)

        # 门控分支
        z_part = tf.nn.silu(z_part)

        # 局部卷积 mixing
        x_part = self.dwconv(x_part)
        x_part = tf.nn.silu(x_part)

        # 近似状态空间混合
        s = self.x_proj(x_part)
        s = self.state_proj(s)

        # 门控融合
        x = s * z_part

        x = self.out_proj(x)
        x = self.drop(x, training=training)

        return residual + x

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout


class TaskAdapter(layers.Layer):
    def __init__(self, input_dim, bottleneck_dim=128, dropout=0.1, **kwargs):
        super(TaskAdapter, self).__init__(**kwargs)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.down = layers.Dense(bottleneck_dim, activation='relu')
        self.dropout = layers.Dropout(dropout)
        self.up = layers.Dense(input_dim)

    def call(self, x, training=None):
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.dropout(x, training=training)
        x = self.up(x)
        return residual + x

class AttentionPooling1D(layers.Layer):
    def __init__(self, hidden_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(hidden_dim, activation='tanh')
        self.dense2 = layers.Dense(1)

    def call(self, x):
        # x: [B, L, D]
        scores = self.dense2(self.dense1(x))   # [B, L, 1]
        weights = tf.nn.softmax(scores, axis=1)  # 在序列维度归一化
        pooled = tf.reduce_sum(weights * x, axis=1)  # [B, D]
        return pooled

def build_protein_classifier_mamba(seq_len=121, feature_dim=1024, alpha_out1=None, alpha_out2=None,gamma=2.0):

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
    outputs_1 = Dense(128, activation='relu')(outputs_1)
    outputs_1 = Dropout(0.1)(outputs_1)
    outputs_1 = Dense(64, activation='relu')(outputs_1)
    outputs_1 = Dropout(0.1)(outputs_1)
    outputs_1 = Dense(7, activation='softmax', name='outputs_1')(outputs_1)

    outputs_2 = layers.Dropout(0.1)(switch_f)
    outputs_2 = Dense(512, activation='relu')(outputs_2)
    # outputs_2 = layers.BatchNormalization()(outputs_2)
    outputs_2 = Dropout(0.1)(outputs_2)
    outputs_2 = Dense(256, activation='relu')(outputs_2)
    outputs_2 = Dropout(0.1)(outputs_2)
    outputs_2 = Dense(128, activation='relu')(outputs_2)
    outputs_2 = Dropout(0.1)(outputs_2)
    outputs_2 = Dense(64, activation='relu')(outputs_2)
    outputs_2 = Dropout(0.1)(outputs_2)
    outputs_2 = Dense(9, activation='softmax', name='outputs_2')(outputs_2)

    model = keras.Model(inputs, [outputs_1, outputs_2])

    model.compile(loss=losses, optimizer=tf.optimizers.Adam(0.0001), metrics=["accuracy"])

    return model




def build_protein_classifier_mamba_first(seq_len=121, feature_dim=1024, alpha_out1=None, alpha_out2=None,
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

    model.compile(loss=focal_loss_first(), optimizer=tf.optimizers.Adam(0.0001), metrics=['accuracy'])  #"BinaryCrossentropy"  loss=focal_loss()

    return model



def build_protein_classifier_mamba_adapter(seq_len=121,
                                           feature_dim=1024,
                                           alpha_out1=None,
                                           alpha_out2=None,
                                           gamma=2.0):
    # 输入层: [Batch, 121, 1024]
    inputs = layers.Input(shape=(seq_len, feature_dim), name="protein_features")

    # ======================
    # 1. Shared Encoder
    # ======================
    x = layers.Dense(1024, activation='relu')(inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    x = MambaBlock(d_model=1024, d_state=128, d_conv=4, expand=2, dropout=0.1)(x)
    x = MambaBlock(d_model=1024, d_state=128, d_conv=4, expand=2, dropout=0.1)(x)

    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    combined = layers.Concatenate(name="shared_pooled_features")([avg_pool, max_pool])   # [2048]

    shared_feat = layers.Dense(1024, activation='relu', name="shared_dense")(combined)
    shared_feat = layers.Dropout(0.2)(shared_feat)
    shared_feat = layers.LayerNormalization(epsilon=1e-6, name="shared_norm")(shared_feat)

    # ======================
    # 2. Task-specific Adapter
    # ======================
    task1_feat = TaskAdapter(input_dim=1024, bottleneck_dim=128, dropout=0.1, name="task1_adapter")(shared_feat)
    task2_feat = TaskAdapter(input_dim=1024, bottleneck_dim=128, dropout=0.1, name="task2_adapter")(shared_feat)


    outputs_1 = layers.Dense(512, activation='relu')(task1_feat)
    outputs_1 = Dropout(0.1)(outputs_1)
    outputs_1 = layers.Dense(128, activation='relu')(outputs_1)
    outputs_1 = Dropout(0.1)(outputs_1)
    outputs_1 = layers.Dense(64, activation='relu')(outputs_1)
    outputs_1 = Dropout(0.1)(outputs_1)
    outputs_1 = layers.Dense(7, activation='softmax', name='outputs_1')(outputs_1)

    outputs_2 = layers.Dense(512, activation='relu')(task2_feat)
    outputs_2 = Dropout(0.1)(outputs_2)
    outputs_2 = layers.Dense(128, activation='relu')(outputs_2)
    outputs_2 = Dropout(0.1)(outputs_2)
    outputs_2 = layers.Dense(64, activation='relu')(outputs_2)
    outputs_2 = Dropout(0.1)(outputs_2)
    outputs_2 = layers.Dense(9, activation='softmax', name='outputs_2')(outputs_2)

    model = keras.Model(inputs, [outputs_1, outputs_2])
    model.compile(loss=losses, optimizer=tf.optimizers.Adam(0.0001), metrics=["accuracy"])

    return model


alpha_family = [0.16,0.079,0.14,0.34,0.14,0.04,0.01]
alpha_virus = [0.17,0.04,0.02,0.2,0.08,0.164,0.1,0.08,0.01]
tokenizer = BertTokenizer.from_pretrained("./model/model/1kmer_model/1kmer_model/vocab.txt")
losses = {'outputs_1': focal_loss(alpha_family), 'outputs_2':  focal_loss(alpha_virus)}

amino_acids = "XACDEFGHIKLMNPQRSTVWY"
protein_dict = dict((c, i) for i, c in enumerate(amino_acids))

# label_family
label_family = pd.read_csv("../dataset/main dataset/second/label_Family.csv", header=None)
label_family = np.array(pd.DataFrame(label_family))

# label_virus
label_virus = pd.read_csv("../dataset/main dataset/second/label_Virus.csv", header=None)
label_virus = np.array(pd.DataFrame(label_virus))

encoder = LabelEncoder()
label_family = encoder.fit_transform(label_family)
label_virus = encoder.fit_transform(label_virus)
label_family = label_family.reshape((2347, 1))
label_virus = label_virus.reshape((2347, 1))



seq_length = 121

file = open('../dataset/main dataset/second/secondStage.faa', encoding="utf-8")
all_line = file.readlines()
fasta = []
for i in range(len(all_line)):
    if i % 2 == 1:
        fasta.append(all_line[i][0:-1])

list2 = [i.ljust(121, 'O') for i in fasta]
temp = ' '
list3 = []
for y in fasta:
    for i in range(len(y)):
        temp = temp + y[i] + ' '
    list3.append(temp)
    temp = ' '

token = tokenizer(list3, return_tensors='tf', padding=True, truncation=True)
input_data = np.array(token["input_ids"])
input_mask = np.array(token["attention_mask"])
config1 = PretrainedConfig()
config2 = config1.from_json_file('./bert-base-uncased/config.json')

input_data3 = pd.read_csv("./AAC+CKSAAP+PAAC+PHY9_second.csv")  # Amino acid features
input_data3 = pd.DataFrame(input_data3)
input_data3 = np.array(input_data3.values)
transfer=MinMaxScaler(feature_range=[-1,1])
input_data3=transfer.fit_transform(input_data3)



input_data3 = np.load('/media/sdc1/23whsu/ProtT5_subclass/final_embeddings.npy')
print(f"input_data3:{input_data3.shape}")

callback = myCallback()


original_time = time.time()


evaluation = []
evaluation.append(['Original time: {0}'.format(original_time)])
data = []
end_time = 0
file_dir = '/media/sdc1/23whsu/model_secondstage{0},{1}'.format(time.strftime("%Y-%m-%d", time.localtime()), original_time)
os.makedirs(file_dir)

mean_Accuracy_1 = []
mean_MacroP_1 = []
mean_MacroR_1 = []
mean_MacroF_1 = []

mean_Accuracy_2 = []
mean_MacroP_2 = []
mean_MacroR_2 = []
mean_MacroF_2 = []

alpha_out1 = compute_alpha_from_sparse_labels(label_family, 7)
alpha_out2 = compute_alpha_from_sparse_labels(label_virus, 9)
# losses = {'outputs_1': focal_loss(alpha_out1), 'outputs_2':  focal_loss(alpha_out2)}
print(alpha_out1)
print(alpha_out2)
input_data_pre = np.load('/media/sdc1/23whsu/ProtT5_AVP_FFMAVP/final_embeddings.npy')
label_pos = np.ones((2762, 1), dtype=int)
label_neg = np.zeros((10089, 1), dtype=int)
label = np.append(label_pos, label_neg)
for i in range(1):
    model_pre = build_protein_classifier_mamba_first()
    label = np_utils.to_categorical(label)
    model_pre.fit(input_data_pre, label, validation_split=0.2,
                epochs=15, batch_size=64)
    model_pre.save_weights("pretrained_weights6.h5")

    model = build_protein_classifier_mamba(alpha_out1=alpha_out1,alpha_out2 = alpha_out2)
    model.summary()
    model.load_weights("pretrained_weights6.h5", by_name=True, skip_mismatch=True)

    evaluation.append((['model_trainable_weights is'], ['{0}'.format(len(model.trainable_weights))]))
    seed = i + 520

    train_set,test_set,train_set2, test_set2, train_set3, test_set3,label_train_family, label_test_family,label_train_virus, label_test_virus\
    = train_test_split(input_data, input_mask, input_data3, label_family, label_virus,test_size=0.2,
                                                                    train_size=0.8, random_state=seed, shuffle=True)

    fold_time = 0
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=0o425)
    start_time = time.time()
    for train, test in kfold.split(train_set, label_train_family):
        evaluation.append(['model{0}_{1} and start time:{2}'.format(i, fold_time, start_time)])

        label_train_family_1= to_one_hot(label_train_family, dimension=7)  

        label_train_virus_1 = to_one_hot(label_train_virus, dimension=9)
        
        model.fit(train_set3[train], [label_train_family_1[train],label_train_virus_1[train]],
                    validation_data=( train_set3[test], [label_train_family_1[test], label_train_virus_1[test]]), epochs=20, batch_size=64,
                    callbacks=[callback])

        evaluation.append(['epoch', 'acc_1', 'acc_2', 'loss', 'val_acc_1', 'val_acc_2', 'val_loss'])
        data = callback.get_data()
        for y in range(len(data)):
            evaluation.append(data[y])
        callback.reset_data()
        fold_time = fold_time + 1

    model.save_weights('{2}/model{0}_{1}.h5'.format(i, fold_time, file_dir))
    end_time = time.time()
    evaluation.append(['model{0} end time:{1},and model cost :{2}'.format(i, end_time, end_time - start_time)])

    # family data and label
    x_test_family_1= []
    x_test_family_2= []
    x_test_family_3 = []
    y_test_family = []
    for x, y in enumerate(label_test_family):
        if y != 6:
            x_test_family_1.append(test_set[x])
            x_test_family_2.append(test_set2[x])
            x_test_family_3.append(test_set3[x])
            y_test_family.append(label_test_family[x])
    # features
    x_test_family_1 = np.array(x_test_family_1)
    x_test_family_2 = np.array(x_test_family_2)
    x_test_family_3 = np.array(x_test_family_3)
    y_test_family = np.array(y_test_family)
    y_test_family = to_one_hot(y_test_family, dimension=6)



    x_test_virus_1 = []
    x_test_virus_2 = []
    x_test_virus_3 = []
    y_test_virus = []
    for x, y in enumerate(label_test_virus):
        if y != 8:
            x_test_virus_1.append(test_set[x])
            x_test_virus_2.append(test_set2[x])
            x_test_virus_3.append(test_set3[x])
            y_test_virus.append(label_test_virus[x])
    # features
    x_test_virus_1 = np.array(x_test_virus_1)
    x_test_virus_2 = np.array(x_test_virus_2)
    x_test_virus_3 = np.array(x_test_virus_3)
    y_test_virus = np.array(y_test_virus)
    y_test_virus = to_one_hot(y_test_virus, dimension=8)
    #这里的y_family和y_virus是换评估之前用的，而y_test_family和y_test_virus是换了评估后用的
    y_family = np.array(label_test_family)
    y_family = to_one_hot(y_family, dimension=7)
    # label_virus
    y_virus = np.array(label_test_virus)
    y_virus = to_one_hot(y_virus, dimension=9)

    # ---------------------------------Task 1 Prediction-------------------------------------------
    preds_family = model.predict(x_test_family_3)
    preds_family = preds_family[0][:, :6]
    print("******************************task1********************************")
    Accuracy, MacroP, MacroR, MacroF = calculate_performace(y_test_family, preds_family)
    print('model%d,Accuracy=%f,MacroP=%f,MacroR=%f,MacroF=%f' % (i, Accuracy, MacroP, MacroR, MacroF))
    evaluation.append(['task1 prediction'])
    evaluation.append(['Accuracy', 'MacroP', 'MacroR', 'MacroF'])
    evaluation.append([str(Accuracy), str(MacroP), str(MacroR), str(MacroF)])
    evaluation.append('\n')
    mean_Accuracy_1.append(Accuracy)
    mean_MacroP_1.append(MacroP)
    mean_MacroR_1.append(MacroR)
    mean_MacroF_1.append(MacroF)
    # ---------------------------------Task 2 Prediction-------------------------------------------
    preds_virus = model.predict(x_test_virus_3)
    preds_virus = preds_virus[1][:, :8]
    Accuracy, MacroP, MacroR, MacroF = calculate_performace(y_test_virus, preds_virus)
    print("******************************task2********************************")
    print('Accuracy=%f,MacroP=%f,MacroR=%f,MacroF=%f' % (Accuracy, MacroP, MacroR, MacroF))
    evaluation.append(['task2 prediction'])
    evaluation.append(['Accuracy', 'MacroP', 'MacroR', 'MacroF'])
    evaluation.append([str(Accuracy), str(MacroP), str(MacroR), str(MacroF)])
    evaluation.append('\n')
    mean_Accuracy_2.append(Accuracy)
    mean_MacroP_2.append(MacroP)
    mean_MacroR_2.append(MacroR)
    mean_MacroF_2.append(MacroF)

evaluation.append(['task1 mean'])
evaluation.append(['mean_Accuracy', 'mean_MacroP', 'mean_MacroR', 'mean_MacroF'])
evaluation.append(
    [np.mean(mean_Accuracy_1), np.mean(mean_MacroP_1), np.mean(mean_MacroR_1), np.mean(mean_MacroF_1)])

evaluation.append(
    [np.std(mean_Accuracy_1), np.std(mean_MacroP_1), np.std(mean_MacroR_1),np.std(mean_MacroF_1)])

evaluation.append(['task2 mean'])
evaluation.append(['mean_Accuracy', 'mean_MacroP', 'mean_MacroR', 'mean_MacroF'])
evaluation.append(
    [np.mean(mean_Accuracy_2), np.mean(mean_MacroP_2), np.mean(mean_MacroR_2), np.mean(mean_MacroF_2)])

evaluation.append(
    [np.std(mean_Accuracy_2), np.std(mean_MacroP_2), np.std(mean_MacroR_2),np.std(mean_MacroF_2)])


evaluation.append(["total_time:{0}".format(time.time() - original_time)])
evaluation = pd.DataFrame(evaluation)
evaluation.to_csv(r"{0}/evaluation.csv".format(file_dir), header=False, index=False)

# ---------------------------------------------以上是第二阶段只把序列按照氨基酸字典编码成数字的--------------------------------------------------------------