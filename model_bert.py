# coding=utf-8

# @Author  : zhzhx2008
# @Time    : 19-7-11
#
# From:
# https://kexue.fm/archives/6736
# https://github.com/CyberZHG/keras-bert


import os
import warnings

import numpy as np
from keras import Input
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

seed = 2019
np.random.seed(seed)

config_path = './data/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './data/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './data/chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}
with open(dict_path, 'r') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)


# # test
# res = tokenizer.tokenize(u'今天天气不错')
# print(res)  # 输出是 ['[CLS]', u'今', u'天', u'天', u'气', u'不', u'错', '[SEP]']


def get_labels_datas(input_dir):
    datas = []
    labels = []
    label_dirs = os.listdir(input_dir)
    for label_dir in label_dirs:
        txt_names = os.listdir(os.path.join(input_dir, label_dir))
        for txt_name in txt_names:
            with open(os.path.join(input_dir, label_dir, txt_name), 'r') as fin:
                content = fin.readline()  # 只取第一行
                content = content.strip().replace(' ', '')
                datas.append(content)
                labels.append(label_dir)
    return labels, datas


def get_label_id_map(labels):
    labels = set(labels)
    id_label_map = {}
    label_id_map = {}
    for index, label in enumerate(labels):
        id_label_map[index] = label
        label_id_map[label] = index
    return id_label_map, label_id_map


input_dir = './data/THUCNews'
labels, datas = get_labels_datas(input_dir)
id_label_map, label_id_map = get_label_id_map(labels)

labels, labels_test, datas, datas_test = train_test_split(labels, datas, test_size=0.3, shuffle=True, stratify=labels)
labels_train, labels_dev, datas_train, datas_dev = train_test_split(labels, datas, test_size=0.1, shuffle=True, stratify=labels)

maxlen = max([len(x) for x in datas_train])

y_train = [label_id_map.get(x) for x in labels_train]
y_dev = [label_id_map.get(x) for x in labels_dev]
y_test = [label_id_map.get(x) for x in labels_test]

num_classes = len(set(y_train))
y_train_index = to_categorical(y_train, num_classes)
y_dev_index = to_categorical(y_dev, num_classes)
y_test_index = to_categorical(y_test, num_classes)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])


class data_generator:
    def __init__(self, datas, labels, batch_size=32):
        self.datas = datas
        self.labels = labels
        self.batch_size = batch_size
        self.steps = len(self.datas) // self.batch_size
        if len(self.datas) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.datas)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                data = self.datas[i]
                label = self.labels[i].tolist()
                text = data[:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = label
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(num_classes, activation='softmax')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['accuracy']
)
model.summary()

train_D = data_generator(datas_train, y_train_index, batch_size=4)
valid_D = data_generator(datas_dev, y_dev_index, batch_size=4)
test_D = data_generator(datas_test, y_test_index, batch_size=4)

model_weight_file = './model_bert.h5'
model_file = './model_bert.model'
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(model_weight_file, save_best_only=True, save_weights_only=True)
model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=train_D.steps,
    epochs=1,
    validation_data=valid_D.__iter__(),
    validation_steps=valid_D.steps,
    callbacks=[early_stopping, model_checkpoint],
    shuffle=True
)
evaluate = model.evaluate_generator(test_D.__iter__(), steps=test_D.steps)
print('loss value=' + str(evaluate[0]))
print('metrics value=' + str(evaluate[1]))

# loss value=0.5063543835625289
# metrics value=0.8571428571428571
