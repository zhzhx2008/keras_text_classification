# coding=utf-8

# @Author  : zhzhx2008
# @Time    : 18-10-8


import os
import warnings

import jieba
import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
                content = fin.readline()# 只取第一行
                content = content.strip().replace(' ', '，')
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
print('y_train\t\tlength=%d' %(len(y_train)))
print('y_dev\t\tlength=%d' %(len(y_dev)))
print('y_test\t\tlength=%d' %(len(y_test)))
print()

# sklearn extract feature
# feature1: count(csr_matrix)
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1))# token_pattern must remove \w, or single char not counted
vectorizer.fit(datas_word_train)
x_train_count = vectorizer.transform(datas_word_train)
x_dev_count = vectorizer.transform(datas_word_dev)
x_test_count = vectorizer.transform(datas_word_test)
print("x_train_count\t\tshape=(%s, %s)" %(x_train_count.shape[0], x_train_count.shape[1]))
print("x_dev_count\t\tshape=(%s, %s)" %(x_dev_count.shape[0], x_dev_count.shape[1]))
print("x_test_count\t\tshape=(%s, %s)" %(x_test_count.shape[0], x_test_count.shape[1]))
print()

# feature2: binary(csr_matrix)
x_train_binary = x_train_count.copy()
x_dev_binary = x_dev_count.copy()
x_test_binary = x_test_count.copy()
x_train_binary[x_train_binary > 0] = 1.0
x_dev_binary[x_dev_binary > 0] = 1.0
x_test_binary[x_test_binary > 0] = 1.0
print("x_train_binary\t\tshape=(%s, %s)" %(x_train_binary.shape[0], x_train_binary.shape[1]))
print("x_dev_binary\t\tshape=(%s, %s)" %(x_dev_binary.shape[0], x_dev_binary.shape[1]))
print("x_test_binary\t\tshape=(%s, %s)" %(x_test_binary.shape[0], x_test_binary.shape[1]))
print()

# feature3: tf-idf(csr_matrix)
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1))
vectorizer.fit(datas_word_train)
x_train_tfidf = vectorizer.transform(datas_word_train)
x_dev_tfidf = vectorizer.transform(datas_word_dev)
x_test_tfidf = vectorizer.transform(datas_word_test)
print("x_train_tfidf\t\tshape=(%s, %s)" %(x_train_tfidf.shape[0], x_train_tfidf.shape[1]))
print("x_dev_tfidf\t\tshape=(%s, %s)" %(x_dev_tfidf.shape[0], x_dev_tfidf.shape[1]))
print("x_test_tfidf\t\tshape=(%s, %s)" %(x_test_tfidf.shape[0], x_test_tfidf.shape[1]))
print()

# keras extract feature
tokenizer = Tokenizer()
tokenizer.fit_on_texts(datas_word_train)
# feature1: count
x_train_count = tokenizer.texts_to_matrix(datas_word_train, mode='count')
x_dev_count = tokenizer.texts_to_matrix(datas_word_dev, mode='count')
x_test_count = tokenizer.texts_to_matrix(datas_word_test, mode='count')
print("x_train_count\t\tshape=(%s, %s)" %(x_train_count.shape[0], x_train_count.shape[1]))
print("x_dev_count\t\tshape=(%s, %s)" %(x_dev_count.shape[0], x_dev_count.shape[1]))
print("x_test_count\t\tshape=(%s, %s)" %(x_test_count.shape[0], x_test_count.shape[1]))
print()

# feature2: binary
x_train_binary = tokenizer.texts_to_matrix(datas_word_train, mode='binary')
x_dev_binary = tokenizer.texts_to_matrix(datas_word_dev, mode='binary')
x_test_binary = tokenizer.texts_to_matrix(datas_word_test, mode='binary')
print("x_train_binary\t\tshape=(%s, %s)" %(x_train_binary.shape[0], x_train_binary.shape[1]))
print("x_dev_binary\t\tshape=(%s, %s)" %(x_dev_binary.shape[0], x_dev_binary.shape[1]))
print("x_test_binary\t\tshape=(%s, %s)" %(x_test_binary.shape[0], x_test_binary.shape[1]))
print()

# feature3: tf-idf
x_train_tfidf = tokenizer.texts_to_matrix(datas_word_train, mode='tfidf')
x_dev_tfidf = tokenizer.texts_to_matrix(datas_word_dev, mode='tfidf')
x_test_tfidf = tokenizer.texts_to_matrix(datas_word_test, mode='tfidf')
print("x_train_tfidf\t\tshape=(%s, %s)" %(x_train_tfidf.shape[0], x_train_tfidf.shape[1]))
print("x_dev_tfidf\t\tshape=(%s, %s)" %(x_dev_tfidf.shape[0], x_dev_tfidf.shape[1]))
print("x_test_tfidf\t\tshape=(%s, %s)" %(x_test_tfidf.shape[0], x_test_tfidf.shape[1]))
print()

# feature4: freq
x_train_freq = tokenizer.texts_to_matrix(datas_word_train, mode='freq')
x_dev_freq = tokenizer.texts_to_matrix(datas_word_dev, mode='freq')
x_test_freq = tokenizer.texts_to_matrix(datas_word_test, mode='freq')
print("x_train_freq\t\tshape=(%s, %s)" %(x_train_freq.shape[0], x_train_freq.shape[1]))
print("x_dev_freq\t\tshape=(%s, %s)" %(x_dev_freq.shape[0], x_dev_freq.shape[1]))
print("x_test_freq\t\tshape=(%s, %s)" %(x_test_freq.shape[0], x_test_freq.shape[1]))
print()

# feature5: word index for deep learning
x_train_word_index = tokenizer.texts_to_sequences(datas_word_train)
x_dev_word_index = tokenizer.texts_to_sequences(datas_word_dev)
x_test_word_index = tokenizer.texts_to_sequences(datas_word_test)
max_word_length = max([len(x) for x in x_train_word_index])
x_train_word_index = sequence.pad_sequences(x_train_word_index, maxlen=max_word_length)
x_dev_word_index = sequence.pad_sequences(x_dev_word_index, maxlen=max_word_length)
x_test_word_index = sequence.pad_sequences(x_test_word_index, maxlen=max_word_length)
print("x_train_index\t\tshape=(%s, %s)" %(x_train_word_index.shape[0], x_train_word_index.shape[1]))
print("x_dev_index\t\tshape=(%s, %s)" %(x_dev_word_index.shape[0], x_dev_word_index.shape[1]))
print("x_test_index\t\tshape=(%s, %s)" %(x_test_word_index.shape[0], x_test_word_index.shape[1]))
