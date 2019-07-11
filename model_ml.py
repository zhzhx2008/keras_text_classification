# coding=utf-8

# @Author  : zhzhx2008
# @Time    : 18-10-8


import os
import warnings

import jieba
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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

# sklearn extract feature
# feature2: binary(csr_matrix)
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1,1))# token_pattern must remove \w, or single char not counted
vectorizer.fit(datas_word_train)

x_train_count = vectorizer.transform(datas_word_train)
x_dev_count = vectorizer.transform(datas_word_dev)
x_test_count = vectorizer.transform(datas_word_test)

x_train_binary = x_train_count.copy()
x_dev_binary = x_dev_count.copy()
x_test_binary = x_test_count.copy()
x_train_binary[x_train_binary > 0] = 1
x_dev_binary[x_dev_binary > 0] = 1
x_test_binary[x_test_binary > 0] = 1

x_train_binary = x_train_binary.astype(np.float32)
x_dev_binary = x_dev_binary.astype(np.float32)
x_test_binary = x_test_binary.astype(np.float32)

clf_dict = {
    'LogisticRegression': LogisticRegression(),
    'SGDClassifier': SGDClassifier(),
    'Perceptron': Perceptron(),
    'MLPClassifier': MLPClassifier(),
    # 'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),  # require dense numpy array
    # 'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),  # require dense numpy array
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(),
    'LinearSVC': LinearSVC(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    # 'GaussianNB': GaussianNB(),  # require dense numpy array
    'MultinomialNB': MultinomialNB(),
    'BernoulliNB': BernoulliNB(),
    'XGBClassifier': XGBClassifier(),
    'LGBMClassifier': LGBMClassifier()
}
for k, v in clf_dict.items():
    clf = v
    clf.fit(x_train_binary, y_train)
    y_pred = clf.predict(x_test_binary)
    y_pred_int = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, y_pred_int)
    print('accuracy of ' + k + '=' + str(accuracy))



# accuracy of LogisticRegression=0.6666666666666666
# accuracy of SGDClassifier=0.6111111111111112
# accuracy of Perceptron=0.6349206349206349
# accuracy of MLPClassifier=0.6587301587301587
# accuracy of KNeighborsClassifier=0.38095238095238093
# accuracy of SVC=0.29365079365079366
# accuracy of LinearSVC=0.6746031746031746
# accuracy of DecisionTreeClassifier=0.5238095238095238
# accuracy of RandomForestClassifier=0.5079365079365079
# accuracy of ExtraTreesClassifier=0.5555555555555556
# accuracy of AdaBoostClassifier=0.4603174603174603
# accuracy of GradientBoostingClassifier=0.5793650793650794
# accuracy of MultinomialNB=0.626984126984127
# accuracy of BernoulliNB=0.42063492063492064
# accuracy of XGBClassifier=0.5714285714285714
# accuracy of LGBMClassifier=0.3968253968253968