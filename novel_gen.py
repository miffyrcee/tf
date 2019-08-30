import os
from random import choice

import tensorflow as tf
import tensorflow_datasets as tfds

BUFFER_SIZE = 100
BATCH_SIZE = 100
TAKE_SIZE = 100
FILE_NAMES = '002.txt'
# with open(FILE_NAMES, 'r') as f:
#     with open('003.txt', 'w') as t:
#         for word in f.read():
#             t.write(word + ' ')

text_raw_ds = tf.data.TextLineDataset('003.txt')

tokenizer = tfds.features.text.Tokenizer()

vocab_list = set()
for ex in text_raw_ds:
    vocab_list.update(tokenizer.tokenize(ex.numpy()))

vocab_size = len(vocab_list)
encoder = tfds.features.text.TokenTextEncoder(vocab_list)
for _ in range(1):
    text_ds = text_raw_ds.map(lambda x: (x, 1))  #labeler


def encode(raw, label):
    return encoder.encode(raw.numpy()), tf.cast(label, tf.int64)


text_ds = text_ds.map(lambda x, label: tf.py_function(
    encode, inp=[x, label], Tout=[tf.int64, tf.int64]))
train_data = text_ds.skip(10).shuffle(10)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))
test_data = text_ds.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))
vocab_size += 1
for ex in test_data.take(1):
    print(ex)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

for units in [64, 64]:
    model.add(tf.keras.layers.Dense(units, activation='relu'))

# 输出层。第一个参数是标签个数。
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, epochs=3, validation_data=test_data)
