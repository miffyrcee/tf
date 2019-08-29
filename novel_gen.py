import os

import tensorflow as tf
import tensorflow_datasets as tfds

FILE_NAMES = '/home/miffyrcee/Downloads/002.txt'


def labeler(example, index):
    return example, tf.cast(index, tf.int64)


text_raw_ds = tf.data.TextLineDataset('/home/miffyrcee/Downloads/002.txt')

tokenizer = tfds.features.text.Tokenizer()
vocab_list = set()
for ex in text_raw_ds:
    vocab_list.update(tokenizer.tokenize(ex.numpy()))
vocab_size = vocab_list.__len__()

encoder = tfds.features.text.TokenTextEncoder(vocab_list)

for _ in range(1):
    text_ds = text_raw_ds.map(lambda x: (x, tf.cast(1, tf.int64)))  #labeler


def encode(raw, label):
    return encoder.encode(raw.numpy()), label


text_ds = text_ds.map(lambda raw, label: tf.py_function(
    encode, inp=[raw, label], Tout=(tf.int64, tf.int64)))

train_ds = text_ds.skip(100).shuffle(1000)
train_ds = train_ds.padded_batch(1000, padded_shapes=([-1], []))

test_ds = text_ds.take(100).shuffle(1000)
text_ds = text_ds.padded_batch(1000, padded_shapes=([-1], []))


class novel_model(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.em1 = tf.keras.layers.Embedding(vocab_size + 1, 64)
        self.lsm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        self.d1 = tf.keras.layers.Dense(64, activation='relu')
        self.do = tf.keras.layers.Dense(1, activation='softmax')

    def call(self, x):
        self.em1(x)
        self.lsm1(x)
        self.d1(x)
        self.d1(x)
        return self.do(x)


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
model.fit(train_ds, epochs=3, validation_data=test_ds)
