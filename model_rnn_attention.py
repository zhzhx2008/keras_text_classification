# coding=utf-8

# @Author  : zhzhx2008
# @Time    : 18-10-9

import os
import warnings

import jieba
import numpy as np
from keras import Input
from keras import Model
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.topology import Layer
from keras.layers import Dropout, Bidirectional
from keras.layers import Embedding, Dense
from keras.layers import LSTM, SpatialDropout1D
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


# 《Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems》
# [https://arxiv.org/abs/1512.08756]
# https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print(weighted_input.shape)
        # return weighted_input
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[1],  self.features_dim
        return input_shape[0], self.features_dim


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
# feature5: word index for deep learning
x_train_word_index = tokenizer.texts_to_sequences(datas_word_train)
x_dev_word_index = tokenizer.texts_to_sequences(datas_word_dev)
x_test_word_index = tokenizer.texts_to_sequences(datas_word_test)

max_word_length = max([len(x) for x in x_train_word_index])
x_train_word_index = pad_sequences(x_train_word_index, maxlen=max_word_length)
x_dev_word_index = pad_sequences(x_dev_word_index, maxlen=max_word_length)
x_test_word_index = pad_sequences(x_test_word_index, maxlen=max_word_length)

input = Input(shape=(max_word_length,))
embedding = Embedding(len(tokenizer.word_index) + 1, 128)(input)
embedding = SpatialDropout1D(0.2)(embedding)

# rnn = SimpleRNN(100, return_sequences=True)(embedding)
# rnn = Attention(max_word_length)(rnn)

# rnn = Bidirectional(SimpleRNN(100, return_sequences=True))(embedding)
# rnn = Attention(max_word_length)(rnn)

# rnn = GRU(100, return_sequences=True)(embedding)
# rnn = Attention(max_word_length)(rnn)

# rnn = Bidirectional(GRU(100, return_sequences=True))(embedding)
# rnn = Attention(max_word_length)(rnn)

# rnn = CuDNNGRU(100, return_sequences=True)(embedding)
# rnn = Attention(max_word_length)(rnn)

# rnn = Bidirectional(CuDNNGRU(100, return_sequences=True))(embedding)
# rnn = Attention(max_word_length)(rnn)

# rnn = LSTM(100, return_sequences=True)(embedding)
# rnn = Attention(max_word_length)(rnn)

rnn = Bidirectional(LSTM(100, return_sequences=True))(embedding)
rnn = Attention(max_word_length)(rnn)  # metrics value=0.38647342980771826
# rnn = GlobalMaxPool1D()(rnn)# 0.33816425149567464
# rnn = GlobalAvgPool1D()(rnn)# 0.20772946881499268
# rnn = Flatten()(rnn) # 0.3140096618357488
# rnn = concatenate([GlobalMaxPool1D()(rnn), GlobalAvgPool1D()(rnn)])# 0.24396135280097742

# rnn = CuDNNLSTM(100, return_sequences=True)(embedding)
# rnn = Attention(max_word_length)(rnn)

# rnn = Bidirectional(CuDNNLSTM(100, return_sequences=True))(embedding)
# rnn = Attention(max_word_length)(rnn)

drop = Dropout(0.2)(rnn)
output = Dense(num_classes, activation='softmax')(drop)
model = Model(inputs=input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model_weight_file = './model_rnn_attention.h5'
model_file = './model_rnn_attention.model'
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(model_weight_file, save_best_only=True, save_weights_only=True)
model.fit(x_train_word_index,
          y_train_index,
          batch_size=32,
          epochs=1000,
          verbose=2,
          callbacks=[early_stopping, model_checkpoint],
          validation_split=0.2,
          shuffle=True)

model.load_weights(model_weight_file)
model.save(model_file)
evaluate = model.evaluate(x_test_word_index, y_test_index, batch_size=32, verbose=2)
print('loss value=' + str(evaluate[0]))
print('metrics value=' + str(evaluate[1]))

# loss value=2.048406752329024
# metrics value=0.5555555560286083