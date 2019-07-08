# coding=utf-8

# @Author  : zhzhx2008
# @Time    : 18-10-9
import warnings

import numpy as np
from keras import Input
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Dense, Dropout, GlobalMaxPool1D, Permute, Reshape, merge
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.datasets import fetch_20newsgroups


def attention_3d_block(x):
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = int(x.shape[1])
    input_dim = int(x.shape[2])
    a = Permute((2, 1))(x)
    a = Reshape((input_dim, time_steps))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    # if SINGLE_ATTENTION_VECTOR:
    #     a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    #     a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([x, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


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
attention_layer = attention_3d_block(embedding)
global_max_pool = GlobalMaxPool1D()(attention_layer)
drop = Dropout(0.5)(global_max_pool)
output = Dense(num_classes, activation='softmax')(drop)
model = Model(inputs=input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model_weight_file = './model_attention.h5'
model_file = './model_attention.model'
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
