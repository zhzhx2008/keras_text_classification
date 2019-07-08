# coding=utf-8

# @Author  : zhzhx2008
# @Time    : 18-10-13
import warnings

import numpy as np
from keras import Input
from keras import Model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Dense, Conv1D, Dropout
from keras.layers import SpatialDropout1D, MaxPooling1D, GlobalMaxPooling1D, add, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
# from keras.utils import plot_model
from keras.utils import to_categorical
from sklearn.datasets import fetch_20newsgroups


def block(x, pool_size=3, strides=2, kernel_size=3):
    filters = x.shape[2].value
    x = MaxPooling1D(pool_size=pool_size, strides=strides)(x)
    x_origin = x
    x = Activation('relu')(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='linear')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='linear')(x)
    x = add([x_origin, x])
    return x


warnings.filterwarnings("ignore")

seed = 1234567
np.random.seed(seed)

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=seed, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=seed, remove=('headers', 'footers', 'quotes'))

# keras extract feature
tokenizer = Tokenizer()
tokenizer.fit_on_texts(newsgroups_train.data)
# feature5: word index for deep learning
x_train_word_index = tokenizer.texts_to_sequences(newsgroups_train.data)
x_test_word_index = tokenizer.texts_to_sequences(newsgroups_test.data)
max_word_length = max([len(x) for x in x_train_word_index])
x_train_word_index = pad_sequences(x_train_word_index, maxlen=max_word_length)
x_test_word_index = pad_sequences(x_test_word_index, maxlen=max_word_length)

y_train = newsgroups_train.target
y_test = newsgroups_test.target
num_classes = len(set(y_train))
y_train_index = to_categorical(y_train, num_classes)
y_test_index = to_categorical(y_test, num_classes)

input = Input(shape=(max_word_length,))
embedding = Embedding(len(tokenizer.word_index) + 1, 128)(input)
embedding = SpatialDropout1D(0.2)(embedding)
conv = Conv1D(64, kernel_size=1, padding='same', activation='relu')(embedding)
block_layer = block(conv)
block_layer = block(block_layer)
block_layer = block(block_layer)
block_layer = block(block_layer)
output = GlobalMaxPooling1D()(block_layer)
output = Dense(256, activation='relu')(output)
output = Dropout(0.5)(output)
output = Dense(num_classes, activation='softmax')(output)
model = Model(input, output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# plot_model(model, show_shapes=True, to_file='model_dpcnn.png')
print(model.summary())

model_weight_file = './model_dpcnn.h5'
model_file = './model_dpcnn.model'
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
