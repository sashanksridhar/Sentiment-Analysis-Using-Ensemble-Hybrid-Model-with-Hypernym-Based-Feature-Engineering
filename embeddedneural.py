import csv

rows = []
rowsx = []
yx = []
y = []
with open("modifiedNoHypernym.csv", 'r', encoding='latin1') as csv1:
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
SEQ_LEN = 80

from keras.preprocessing.sequence import pad_sequences
padded_train = pad_sequences(encoded_train_set, maxlen=SEQ_LEN, padding='post')

train_docs = [list(doc) for doc in padded_train]

from keras.models import Sequential,Model
from keras.layers import  Dense,Input
from keras.layers import LSTM,Embedding,Bidirectional
from keras.optimizers import Adam
input_tensor = Input(shape=(SEQ_LEN,), dtype='int32')
e = Embedding(vocab_size, 300, input_length=SEQ_LEN, trainable=True)(input_tensor)
x = Bidirectional(LSTM(128, return_sequences=True))(e)
x = Bidirectional(LSTM(64, return_sequences=False))(x)
x = Dense(64, activation='relu')(x)
output_tensor = Dense(2, activation='sigmoid')(x)
model = Model(input_tensor, output_tensor)

model.compile(optimizer=Adam(lr=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_train = np.array([np.array(token) for token in train_docs])
model.fit(x_train, ytx, epochs=10,batch_size=512)
model.save("Em10NoHypernym.h5")

rowsx = []
yx = []

with open("testCombinedNoHypernym.csv", 'r', encoding='latin1') as csv1:
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

