# Text Classification Conclusion
## preprocessing
- 纠错
- 繁转简
- 去除标点符号，空格，回车等特殊字符
- 去除停用词
- 去除低频词，高频词
- 大写转小写
- 命名实体识别
- 新词发现
- 分词
- 词性标注
- 句法分析
- 数据扩充
    - 倒序(char or word)
    - 以一定的概率随机增加或者去掉char or word
    - 以一定的概率随机打乱词序或部分词交换位置
    - 翻译成别的语言再翻译回来
    - 同义词替换或者插入
## feature extraction
基本单位：char, word, pos, n-gram
### ML特征
- BOW
    - count
    - binary
    - tf
    - tf-idf
    - Entropy-based term weighting
      (Entropy-based term weighting schemes for text categorization in VSM, 当一个词语集中分布在少数类别中,可以视为“熵”较小,对类别的区分度较高;反之,当一个词语较为均匀地分布在多个类别中,“熵”就较大,对类别区分度较低)
    - 降维
    - n-gram
- 核心词
    - 高频词
    - 主题模型
        - LDA
        - LSA，or LSI
        - pLSA
- 实体
- 正则
- 否定词
- 疑问词
- 句法分析特征
- char length
- word length
- stopwords count
### DL特征
- word embeddings(char, word, pos, n-gram)
- sentence embeddings
- 随机初始化
- 基于预训练的向量
    - google word2vec
    - glove
    - fasttext
    - 自己训练
- 不同方法训练的词向量concatenate（2d or 3d），or avg，or 20%A+80%B，etc
## model
### ML methods
LR,xgboost,lightgbm,MultinomialNB,BernoulliNB,LinearSVC,GBDT,MLP,...
### DL methods
- FastText
- CNN（https://github.com/brightmart/text_classification）
    - 1D
    - 2D
- GRU, LSTM
- Tree-LSTM
- Attention
- Transformer
- Bert
- Capsule
- seq2seq

- DPCNN
- RCNN
- EntNet, DMN
- HAN
- Hybird NN
## ensemble
- bagging
    - avg
    - voting
- boosting
- stacking
## other
- padding: max or avg or most frequency sequence length(如果句子过长，可以取前100，或者后100，或者前50+后50)
- bidirectional, CuDNNLSTM
- spatial dropout
- dropout
- batchnormalization
- 十折交叉验证
- maxPool， avgPool
