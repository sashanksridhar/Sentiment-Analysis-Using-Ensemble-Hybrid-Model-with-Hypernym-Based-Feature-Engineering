import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from os import walk
import csv
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
import pandas as pd
classes = sorted(pd.DataFrame(yx)[0].unique())
label_to_cat = dict()
for i in range(len(classes)):
    dummy = np.zeros((len(classes),), dtype='int8')
    dummy[i] = 1
    label_to_cat[classes[i]] = dummy

cat_to_label = dict()
for k, v in label_to_cat.items():
    cat_to_label[tuple(v)] = k

ytx = np.array([label_to_cat[label] for label in yx])
print(ytx)


from gensim.models import Word2Vec


embeddings = Word2Vec(size=200, min_count=3)
embeddings.build_vocab([sentence for sentence in rowsx])
embeddings.train([sentence for sentence in rowsx],
                 total_examples=embeddings.corpus_count,
                 epochs=embeddings.epochs)
# print(embeddings.wv.most_similar('economy'))

gen_tfidf = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
matrix = gen_tfidf.fit_transform([sentence   for sentence in rowsx])
tfidf_map = dict(zip(gen_tfidf.get_feature_names(), gen_tfidf.idf_))
print(len(tfidf_map))


from sklearn.preprocessing import scale


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


def encode_sentence_lstm(tokens, emb_size):
    vec = np.zeros((6, 200))
    for i, word in enumerate(tokens):
        if i > 5:
            break
        try:
            vec[i] = embeddings.wv[word].reshape((1, emb_size))
        except KeyError:
            continue
    return vec

# x_train = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, rowsx)]))
x_train = np.array([encode_sentence_lstm(ele, 200) for ele in map(lambda x: x, rowsx)])
print(x_train.shape)

from keras.models import Sequential,Model
from keras.layers import  Dense,Input
from keras.layers import LSTM
from keras.optimizers import Adam
# model = Sequential()
# model.add(Dense(256, activation='relu', input_dim=200))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(2048, activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
#
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer=Adam(lr=1e-3, decay=1e-6),loss='binary_crossentropy',metrics=['accuracy'])
# model.fit(x_train, yx, epochs=50, batch_size=512)
# model.save("NewANN50-2epochs.h5")


input_tensor = Input(shape=(6, 200))
x = LSTM(256, return_sequences=True)(input_tensor)
x1 = LSTM(256, return_sequences=True)(x)
x2 = LSTM(256, return_sequences=False)(x1)
x = Dense(64, activation='relu')(x2)
output_tensor = Dense(2, activation='softmax')(x)
model = Model(inputs=[input_tensor], outputs=[output_tensor])
model.compile(optimizer=Adam(lr=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, ytx, epochs=50, batch_size=512)
model.save("LSTM10-50epochs.h5")
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
    if int(yx[i])<=4:
        yx[i] = "0"
    else:
        yx[i] = "1"
ytex = np.array([label_to_cat[label] for label in yx])
# print(ytex)




matrix1 = gen_tfidf.transform([sentence   for sentence in rowsx])
tfidf_map1 = dict(zip(gen_tfidf.get_feature_names(), gen_tfidf.idf_))
print(len(tfidf_map1))

# x_test = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, rowsx)]))
x_test = np.array([encode_sentence_lstm(ele, 200) for ele in map(lambda x: x, rowsx)])
print(x_test.shape)



score = model.evaluate(x_test, yx)
print(score)



