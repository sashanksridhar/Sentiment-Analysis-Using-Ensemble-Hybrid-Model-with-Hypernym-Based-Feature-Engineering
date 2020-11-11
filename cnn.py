import csv

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
        yx[i] = "0"
    else:
        yx[i] = "1"

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

from keras.preprocessing.text import Tokenizer
t = Tokenizer()

t.fit_on_texts(rowsx)
vocab_size = len(t.word_index) + 1
print(vocab_size)
encoded_train_set = t.texts_to_sequences(rowsx)
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300
from keras.preprocessing.sequence import pad_sequences
padded_train = pad_sequences(encoded_train_set, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

train_docs = [list(doc) for doc in padded_train]

from keras.models import Sequential,Model
from keras.layers import Dense, Input, Conv1D, GlobalMaxPooling1D, Dropout
from keras.layers import LSTM,Embedding,Bidirectional,concatenate
from keras.optimizers import Adam
train_word_index = t.word_index
train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))

embeddings = train_embedding_weights
max_sequence_length = MAX_SEQUENCE_LENGTH
num_words =vocab_size
embedding_dim = EMBEDDING_DIM
labels_index = 2


embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=False)

sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
convs = []
filter_sizes = [2, 3, 4, 5, 6]
for filter_size in filter_sizes:
    l_conv = Conv1D(filters=200,
                    kernel_size=filter_size,
                    activation='relu')(embedded_sequences)
    l_pool = GlobalMaxPooling1D()(l_conv)
    convs.append(l_pool)
l_merge = concatenate(convs, axis=1)
x = Dropout(0.1)(l_merge)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
preds = Dense(labels_index, activation='sigmoid')(x)
model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])


x_train = np.array([np.array(token) for token in train_docs])
model.fit(x_train, ytx, epochs=50, batch_size=512)
model.save("CNN50.h5")

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
print(ytex)

