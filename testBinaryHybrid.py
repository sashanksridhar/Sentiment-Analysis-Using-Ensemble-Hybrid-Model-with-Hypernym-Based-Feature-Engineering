import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import scale

from sklearn.metrics import classification_report, confusion_matrix
import csv
from keras.models import load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter

rowsx = []
yx = []

with open("E:\\sentimentanalysis\\BiLSTMtrainDataFull.csv", 'r', encoding='latin1') as csv1:
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
    # print(rows1)

        rowsx.append(rows1)

        # print(len(rows))

for i in range(0,len(yx)):
    if int(yx[i])<=4:
        yx[i] = 0
    else:
        yx[i] = 1


rowsx1 = []


with open("E:\\sentimentanalysis\\ANNtrainDataFull.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(1, len(row)):

            for j in row[i].split("\n"):

                    rows1.append(j)



        del (row[0])
    # print(rows1)

        rowsx1.append(rows1)

t = Tokenizer()

t.fit_on_texts(rowsx)
vocab_size = len(t.word_index) + 1
print(vocab_size)

embeddings = Word2Vec(size=200, min_count=3)
embeddings.build_vocab([sentence for sentence in rowsx1])
embeddings.train([sentence for sentence in rowsx1],
                 total_examples=embeddings.corpus_count,
                 epochs=embeddings.epochs)
# print(embeddings.wv.most_similar('economy'))

gen_tfidf = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
matrix = gen_tfidf.fit_transform([sentence   for sentence in rowsx1])
tfidf_map = dict(zip(gen_tfidf.get_feature_names(), gen_tfidf.idf_))
print(len(tfidf_map))





def encode_sentence(tokens, emb_size):
    _vector = np.zeros((1, emb_size))
    length = 0
    for word in tokens:
        try:
            _vector += embeddings.wv[word].reshape((1, emb_size)) * tfidf_map[word]
            length += 1
        except KeyError:
            continue
        break

    if length > 0:
        _vector /= length

    return _vector


model = load_model("Hybrid10-2.h5")

rowsx_t_1 = []
yx_1 = []

with open("E:\\sentimentanalysis\\BiLSTMtrainDataTest.csv", 'r', encoding='latin1') as csv1:
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


        yx_1.append(row[0])
        del (row[0])
    # print(rows1)

        rowsx_t_1.append(rows1)

        # print(len(rows))

for i in range(0,len(yx_1)):
    if int(yx_1[i])<=4:
        yx_1[i] = 0
    else:
        yx_1[i] = 1


rowsx_t_2 = []


with open("E:\\sentimentanalysis\\ANNtrainDataTest.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(1, len(row)):

            for j in row[i].split("\n"):

                    rows1.append(j)



        del (row[0])
    # print(rows1)

        rowsx_t_2.append(rows1)






encoded_train_set = t.texts_to_sequences(rowsx_t_1)
SEQ_LEN = 80


padded_train = pad_sequences(encoded_train_set, maxlen=SEQ_LEN, padding='post')

train_docs = [list(doc) for doc in padded_train]
x_train = np.array([np.array(token) for token in train_docs])
x_train1 = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, rowsx_t_2)]))


count = 0


# score = model.evaluate([x_train,x_train1], yx_1)
# print(score)
pred = []
predicted = model.predict([x_train,x_train1])
print(len(predicted))
for i in range(0,len(yx_1)):
    ytrue = 0
    if predicted[i][0] < 0.5:
        ytrue = 0
    else:
        ytrue = 1
    if ytrue == yx_1[i]:
        count += 1
    pred.append(ytrue)
# for i in range(0,len(yx_1)):
#     # print(x_test[i])
#
#     predicted = model.predict(np.array([x_train[i],]),np.array([x_train1[i],]))
#
#     ytrue = 0
#     if predicted[0]<0.5:
#         ytrue = 0
#     else:
#         ytrue = 1
#     if ytrue == yx_1[i]:
#         count+=1
#     pred.append(ytrue)

print(count)
print((count/len(yx))*100)
print("=== Confusion Matrix ===")
print(confusion_matrix(yx, pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(yx, pred))
print('\n')
import scikitplot
import matplotlib.pyplot as plt
scikitplot.metrics.plot_confusion_matrix(yx, pred)
plt.show()
