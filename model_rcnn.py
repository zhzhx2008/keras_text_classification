# coding=utf-8

# @Author  : zhzhx2008
# @Time    : 18-10-9
# https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier

import warnings

import numpy as np
import keras.backend as K
from keras import Input
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Dense, Conv1D, MaxPool1D, concatenate, Flatten, Dropout, GlobalMaxPool1D, LSTM, TimeDistributed, Lambda, CuDNNLSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.datasets import fetch_20newsgroups

warnings.filterwarnings("ignore")

seed = 1234567
np.random.seed(seed)

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=seed, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=seed, remove=('headers', 'footers', 'quotes'))

# keras extract feature
tokenizer = Tokenizer()
tokenizer.fit_on_texts(newsgroups_train.data)
vocabulary_length = len(tokenizer.word_index)
# feature5: word index for deep learning
x_train_word_index = tokenizer.texts_to_sequences(newsgroups_train.data)
x_test_word_index = tokenizer.texts_to_sequences(newsgroups_test.data)
max_word_length = max([len(x) for x in x_train_word_index])

y_train = newsgroups_train.target
y_test = newsgroups_test.target
num_classes = len(set(y_train))
y_train_index = to_categorical(y_train, num_classes)
y_test_index = to_categorical(y_test, num_classes)

# RCNN
# dict{vocabulary_length}=0,0,0,0,......
x_train_word_index_left = [[vocabulary_length] + x[:-1] for x in x_train_word_index]
x_test_word_index_left = [[vocabulary_length] + x[:-1] for x in x_test_word_index]
x_train_word_index_right = [x[1:] + [vocabulary_length] for x in x_train_word_index]
x_test_word_index_right = [x[1:] + [vocabulary_length] for x in x_test_word_index]

x_train_word_index_left = pad_sequences(x_train_word_index_left, maxlen=max_word_length)
x_test_word_index_left = pad_sequences(x_test_word_index_left, maxlen=max_word_length)
x_train_word_index_right = pad_sequences(x_train_word_index_right, maxlen=max_word_length)
x_test_word_index_right = pad_sequences(x_test_word_index_right, maxlen=max_word_length)
x_train_word_index = pad_sequences(x_train_word_index, maxlen=max_word_length)
x_test_word_index = pad_sequences(x_test_word_index, maxlen=max_word_length)

input = Input(shape=(max_word_length,), dtype='int32')
input_left = Input(shape=(max_word_length,), dtype='int32')
input_right = Input(shape=(max_word_length,), dtype='int32')

embedding = Embedding(vocabulary_length + 1, 100)
embedding_input = embedding(input)
embedding_input_left = embedding(input_left)
embedding_input_right = embedding(input_right)

# SimpleRNN, GRU, or LSTM
forward = CuDNNLSTM(128, return_sequences=True)(embedding_input_left)
backward = CuDNNLSTM(128, return_sequences=True, go_backwards=True)(embedding_input_right)
# Keras returns the output sequences in reverse order.
backward = Lambda(lambda x: K.reverse(x, axes = 1))(backward)
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
          validation_split=0.2,
          shuffle=True)

model.load_weights(model_weight_file)
model.save(model_file)
evaluate = model.evaluate([x_test_word_index, x_test_word_index_left, x_test_word_index_right], y_test_index, batch_size=32, verbose=2)
print('loss value=' + str(evaluate[0]))
print('metrics value=' + str(evaluate[1]))
