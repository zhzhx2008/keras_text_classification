# coding=utf-8

# @Author  : zhzhx2008
# @Time    : 18-10-9

# Reference: https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier

import os
import warnings

import jieba
import keras.backend as K
import numpy as np
from keras import Input
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Dense, concatenate, LSTM, TimeDistributed, Lambda
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

seed = 2019
np.random.seed(seed)


def get_labels_datas(input_dir):
    datas_word = []
    datas_char = []
    labels = []
    label_dirs = os.listdir(input_dir)
    for label_dir in label_dirs:
        txt_names = os.listdir(os.path.join(input_dir, label_dir))
        for txt_name in txt_names:
            with open(os.path.join(input_dir, label_dir, txt_name), 'r') as fin:
                content = fin.readline()  # 只取第一行
                content = content.strip().replace(' ', '')
                datas_word.append(' '.join(jieba.cut(content)))
                datas_char.append(' '.join(list(content)))
                labels.append(label_dir)
    return labels, datas_word, datas_char


def get_label_id_map(labels):
    labels = set(labels)
    id_label_map = {}
    label_id_map = {}
    for index, label in enumerate(labels):
        id_label_map[index] = label
        label_id_map[label] = index
    return id_label_map, label_id_map


input_dir = './data/THUCNews'
labels, datas_word, datas_char = get_labels_datas(input_dir)
id_label_map, label_id_map = get_label_id_map(labels)

labels, labels_test, datas_word, datas_word_test, datas_char, datas_char_test = train_test_split(labels, datas_word, datas_char, test_size=0.3, shuffle=True, stratify=labels)
labels_train, labels_dev, datas_word_train, datas_word_dev, datas_char_train, datas_char_dev = train_test_split(labels, datas_word, datas_char, test_size=0.1, shuffle=True, stratify=labels)

y_train = [label_id_map.get(x) for x in labels_train]
y_dev = [label_id_map.get(x) for x in labels_dev]
y_test = [label_id_map.get(x) for x in labels_test]

num_classes = len(set(y_train))
y_train_index = to_categorical(y_train, num_classes)
y_dev_index = to_categorical(y_dev, num_classes)
y_test_index = to_categorical(y_test, num_classes)

# keras extract feature
tokenizer = Tokenizer()
tokenizer.fit_on_texts(datas_word_train)
vocabulary_length = len(tokenizer.word_index)
# feature5: word index for deep learning
x_train_word_index = tokenizer.texts_to_sequences(datas_word_train)
x_dev_word_index = tokenizer.texts_to_sequences(datas_word_dev)
x_test_word_index = tokenizer.texts_to_sequences(datas_word_test)

# RCNN
# dict{vocabulary_length}=0,0,0,0,......
x_train_word_index_left = [[vocabulary_length + 1] + x[:-1] for x in x_train_word_index]
x_dev_word_index_left = [[vocabulary_length + 1] + x[:-1] for x in x_dev_word_index]
x_test_word_index_left = [[vocabulary_length + 1] + x[:-1] for x in x_test_word_index]
x_train_word_index_right = [x[1:] + [vocabulary_length + 1] for x in x_train_word_index]
x_dev_word_index_right = [x[1:] + [vocabulary_length + 1] for x in x_dev_word_index]
x_test_word_index_right = [x[1:] + [vocabulary_length + 1] for x in x_test_word_index]

max_word_length = max(
    max([len(x) for x in x_train_word_index]),
    max([len(x) for x in x_train_word_index_left]),
    max([len(x) for x in x_train_word_index_right])
)

x_train_word_index_left = pad_sequences(x_train_word_index_left, maxlen=max_word_length)
x_dev_word_index_left = pad_sequences(x_dev_word_index_left, maxlen=max_word_length)
x_test_word_index_left = pad_sequences(x_test_word_index_left, maxlen=max_word_length)
x_train_word_index_right = pad_sequences(x_train_word_index_right, maxlen=max_word_length)
x_dev_word_index_right = pad_sequences(x_dev_word_index_right, maxlen=max_word_length)
x_test_word_index_right = pad_sequences(x_test_word_index_right, maxlen=max_word_length)
x_train_word_index = pad_sequences(x_train_word_index, maxlen=max_word_length)
x_dev_word_index = pad_sequences(x_dev_word_index, maxlen=max_word_length)
x_test_word_index = pad_sequences(x_test_word_index, maxlen=max_word_length)

input = Input(shape=(max_word_length,), dtype='int32')
input_left = Input(shape=(max_word_length,), dtype='int32')
input_right = Input(shape=(max_word_length,), dtype='int32')

embedding = Embedding(vocabulary_length + 1 + 1, 100)
embedding_input = embedding(input)
embedding_input_left = embedding(input_left)
embedding_input_right = embedding(input_right)

# SimpleRNN, GRU(CuDNNGRU), or LSTM(CuDNNLSTM)
forward = LSTM(128, return_sequences=True)(embedding_input_left)
backward = LSTM(128, return_sequences=True, go_backwards=True)(embedding_input_right)
# Keras returns the output sequences in reverse order.
backward = Lambda(lambda x: K.reverse(x, axes=1))(backward)
together = concatenate([forward, embedding_input, backward], axis=2)
# semantic = Conv1D(128, kernel_size = 1, activation = "tanh")(together)
semantic = TimeDistributed(Dense(128, activation='tanh'))(together)
pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
output = Dense(num_classes, activation='softmax')(pool_rnn)
model = Model(inputs=[input, input_left, input_right], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model_weight_file = './model_rcnn.h5'
model_file = './model_rcnn.model'
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(model_weight_file, save_best_only=True, save_weights_only=True)
model.fit([x_train_word_index, x_train_word_index_left, x_train_word_index_right],
          y_train_index,
          batch_size=32,
          epochs=1000,
          verbose=2,
          callbacks=[early_stopping, model_checkpoint],
          validation_data=([x_dev_word_index, x_dev_word_index_left, x_dev_word_index_right], y_dev_index),
          shuffle=True)

model.load_weights(model_weight_file)
model.save(model_file)
evaluate = model.evaluate([x_test_word_index, x_test_word_index_left, x_test_word_index_right], y_test_index, batch_size=32, verbose=2)
print('loss value=' + str(evaluate[0]))
print('metrics value=' + str(evaluate[1]))

# loss value=1.4838370917335388
# metrics value=0.45238095332705786