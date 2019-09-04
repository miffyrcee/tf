import os
from collections.abc import Iterable
from random import choice

import tensorflow as tf
import tensorflow_datasets as tfds

BUFFER_SIZE = 10000
BATCH_SIZE = 64
TAKE_SIZE = 60
FILE_NAMES = '/home/miffyrcee/github/py/002.txt'
text_raw_ds = tf.data.TextLineDataset(FILE_NAMES)
tokenizer = tfds.features.text.Tokenizer()

vocab_list = set()
for ex in text_raw_ds:
    vocab_list.update(tokenizer.tokenize(ex.numpy()))
vocab_size = vocab_list.__len__()

encoder = tfds.features.text.TokenTextEncoder(vocab_list)


def wrapper(line):
    encoder_list = encoder.encode(line.numpy())
    return encoder_list[:-1], encoder_list[1:]


def split_in_out(line):
    return tf.py_function(wrapper, inp=[line], Tout=[tf.int64, tf.int64])


for e in text_raw_ds.take(1):
    print(e)

text_ds = text_raw_ds.batch(BATCH_SIZE)

text_ds = text_raw_ds.map(split_in_out)

text_ds = text_ds.padded_batch(vocab_size,
                               padded_shapes=([-1], [-1]),
                               drop_remainder=True)
for ex, j in text_ds.take(1):
    print(ex)
