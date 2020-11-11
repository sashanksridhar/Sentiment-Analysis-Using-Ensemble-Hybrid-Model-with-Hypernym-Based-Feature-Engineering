import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import csv
from keras.models import load_model
from collections import Counter

rows = []
rowsx = []
yx = []
y = []
with open("modified.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(1, len(row)):

            for j in row[i].split("\n"):
                # s+=j+" "

                    # print(j)
                    rows1.append(j)
                    y.append(row[0])

        yx.append(row[0])
        del (row[0])
    # print(rows1)
        rows.extend(rows1)
        rowsx.append(rows1)

        # print(len(rows))

for i in range(0,len(yx)):
    if int(yx[i])<=4:
        yx[i] = 0
    else:
        yx[i] = 1

import numpy as np

#
#
# from gensim.models import Word2Vec
#
#
# embeddings = Word2Vec(size=200, min_count=3)
# embeddings.build_vocab([sentence for sentence in rowsx])
# embeddings.train([sentence for sentence in rowsx],
#                  total_examples=embeddings.corpus_count,
#                  epochs=embeddings.epochs)
# # print(embeddings.wv.most_similar('economy'))
#
# gen_tfidf = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
# matrix = gen_tfidf.fit_transform([sentence   for sentence in rowsx])
# tfidf_map = dict(zip(gen_tfidf.get_feature_names(), gen_tfidf.idf_))
# print(len(tfidf_map))
#
#
# from sklearn.preprocessing import scale
#
#
# def encode_sentence(tokens, emb_size):
#     _vector = np.zeros((1, emb_size))
#     length = 0
#     for word in tokens:
#         try:
#             _vector += embeddings.wv[word].reshape((1, emb_size)) * tfidf_map[word]
#             length += 1
#         except KeyError:
#             continue
#         break
#
#     if length > 0:
#         _vector /= length
#
#     return _vector
#
#
# def encode_sentence_lstm(tokens, emb_size):
#     vec = np.zeros((6, 200))
#     for i, word in enumerate(tokens):
#         if i > 5:
#             break
#         try:
#             vec[i] = embeddings.wv[word].reshape((1, emb_size))
#         except KeyError:
#             continue
#     return vec
#
# # x_train = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, rowsx)]))
# x_train = np.array([encode_sentence_lstm(ele, 200) for ele in map(lambda x: x, rowsx)])
# print(x_train.shape)

from keras.preprocessing.text import Tokenizer
t = Tokenizer()

t.fit_on_texts(rowsx)
vocab_size = len(t.word_index) + 1
print(vocab_size)
encoded_train_set = t.texts_to_sequences(rowsx)
SEQ_LEN = 50

from keras.preprocessing.sequence import pad_sequences
padded_train = pad_sequences(encoded_train_set, maxlen=SEQ_LEN, padding='post')

train_docs = [list(doc) for doc in padded_train]
model = load_model("NewCNN50-2.h5")

rowsx = []
yx = []

with open("testCombined.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(1, len(row)):

            for j in row[i].split("\n"):
                # s+=j+" "

                    # print(j)
                    rows1.append(j)


        yx.append(row[0])
        del (row[0])

        rowsx.append(rows1)

for i in range(0,len(yx)):
    # print(i)
    # print(yx[i])
    if int(yx[i])<=4:
        yx[i] = 0
    else:
        yx[i] = 1






# # x_test = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, rowsx)]))
# x_test = np.array([encode_sentence_lstm(ele, 200) for ele in map(lambda x: x, rowsx)])
# print(x_test.shape)

encoded_test = t.texts_to_sequences(rowsx)
padded_test = pad_sequences(encoded_test, maxlen=SEQ_LEN, padding='post')
test_docs = [list(doc) for doc in padded_test]
x_test = np.array([np.array(token) for token in test_docs])
count = 0


score = model.evaluate(x_test, yx)
print(score)
pred = []
for i in range(0,len(x_test)):
    # print(x_test[i])

    predicted = model.predict(np.array([x_test[i],]))

    ytrue = 0
    if predicted[0]<0.5:
        ytrue = 0
    else:
        ytrue = 1
    if ytrue == yx[i]:
        count+=1
    pred.append(ytrue)

print(count)
print((count/len(yx))*100)
print("=== Confusion Matrix ===")
print(confusion_matrix(yx, pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(yx, pred))
print('\n')
