# coding=utf-8

# @Author  : zhzhx2008
# @Time    : 18-10-10
# https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py

# author - Richard Liao
# Dec 26 2016
import logging
import re
import warnings

import numpy as np
from keras import backend as K
from keras import initializers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.engine.topology import Layer
from keras.layers import CuDNNGRU
from keras.layers import Dense, Input
from keras.layers import Embedding, Bidirectional, TimeDistributed
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.datasets import fetch_20newsgroups


def clean_str(string):
    string = string.replace('\n', ' ')
    string = string.replace('\\', ' ')
    string = string.replace('\'', ' ')
    string = string.replace('\"', ' ')
    string = string.replace(',', ' ,')
    string = re.sub(' +', ' ', string)
    return string.strip().lower()


def process_documents(documents, max_sentence_number=0, max_word_number=0):
    max_document_number = 0
    need_calculate = False
    if max_sentence_number == 0 and max_word_number == 0:
        need_calculate = True

    big_list = []
    sentence_list = []
    for document in documents:
        max_document_number += 1
        small_list = []
        document = clean_str(document)
        sentences = re.split('[\.|\?|\!]', document)
        sentence_number = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if "" == sentence:
                continue
            small_list.append(sentence)
            sentence_list.append(sentence)
            sentence_number += 1
            words = sentence.split(' ')
            word_number = len(words)
            if need_calculate:
                max_word_number = max(max_word_number, word_number)
        if need_calculate:
            max_sentence_number = max(max_sentence_number, sentence_number)
        big_list.append(small_list)
    return big_list, sentence_list, max_document_number, max_sentence_number, max_word_number


def list_to_matrix(big_list, x, tokenizer, max_sentence_number, max_word_number):
    index = 0
    for small_list in big_list:
        small_list = tokenizer.texts_to_sequences(small_list)
        small_list = pad_sequences(small_list, maxlen=max_word_number)
        pad_sentence_number = max_sentence_number - small_list.shape[0]
        if pad_sentence_number < 0:
            small_list = small_list[0 - pad_sentence_number, :]
        if pad_sentence_number > 0:
            pad_array = np.zeros((pad_sentence_number, max_word_number))
            small_list = np.row_stack([pad_array, small_list])
        x[index] = small_list
        index += 1
    return x


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

seed = 1234567
np.random.seed(seed)

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=seed, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=seed, remove=('headers', 'footers', 'quotes'))

big_list_train, sentence_list, max_document_number_train, max_sentence_number, max_word_number = process_documents(newsgroups_train.data)
max_sentence_number = min(100, max_sentence_number) # to avoid too large
max_word_number = min(50, max_word_number)
logging.info('big_list_train length=%s, max_document_number_train=%s, max_sentence_number=%s, max_word_number=%s', len(big_list_train), max_document_number_train, max_sentence_number, max_word_number)
x_train = np.zeros((max_document_number_train, max_sentence_number, max_word_number))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentence_list)
x_train = list_to_matrix(big_list_train, x_train, tokenizer, max_sentence_number, max_word_number)

big_list_test, sentence_list, max_document_number_test, _, _ = process_documents(newsgroups_test.data, max_sentence_number, max_word_number)
logging.info('big_list_test length=%s, max_document_number_test=%s, max_sentence_number=%s, max_word_number=%s', len(big_list_test), max_document_number_test, max_sentence_number, max_word_number)
x_test = np.zeros((max_document_number_test, max_sentence_number, max_word_number))
x_test = list_to_matrix(big_list_test, x_test, tokenizer, max_sentence_number, max_word_number)

y_train = newsgroups_train.target
y_test = newsgroups_test.target
num_classes = len(set(y_train))
y_train_index = to_categorical(y_train, num_classes)
y_test_index = to_categorical(y_test, num_classes)


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


sentence_input = Input(shape=(max_word_number,), dtype='int32')
sentence_embedding = Embedding(len(tokenizer.word_index) + 1, 128)(sentence_input)
sentence_bi_gru = Bidirectional(CuDNNGRU(64, return_sequences=True))(sentence_embedding)
sentence_output = AttLayer(64)(sentence_bi_gru)
sentence_model = Model(sentence_input, sentence_output)

document_input = Input(shape=(max_sentence_number, max_word_number), dtype='int32')
sentence_encoder = TimeDistributed(sentence_model)(document_input)
document_bi_gru = Bidirectional(CuDNNGRU(64, return_sequences=True))(sentence_encoder)
document_attention = AttLayer(64)(document_bi_gru)
document_output = Dense(num_classes, activation='softmax')(document_attention)
model = Model(document_input, document_output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model_weight_file = './model_han.h5'
model_file = './model_han.model'
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(model_weight_file, save_best_only=True, save_weights_only=True)
model.fit(x_train,
          y_train_index,
          batch_size=32,
          epochs=1000,
          verbose=2,
          callbacks=[early_stopping, model_checkpoint],
          validation_split=0.2,
          shuffle=True)

model.load_weights(model_weight_file)
model.save(model_file)
evaluate = model.evaluate(x_test, y_test_index, batch_size=32, verbose=2)
print('loss value=' + str(evaluate[0]))
print('metrics value=' + str(evaluate[1]))
