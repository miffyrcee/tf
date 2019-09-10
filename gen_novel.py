import os

import tensorflow as tf
import tensorflow_datasets as tfds

BUFFER_SIZE = 10000
BATCH_SIZE = 51
TAKE_SIZE = 30
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


text_ds = text_raw_ds

text_ds = text_ds.map(split_in_out)

text_ds = text_ds.padded_batch(BATCH_SIZE,
                               padded_shapes=([-1], [-1]),
                               drop_remainder=True)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,
                                  embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(vocab_size=vocab_size + 1,
                    embedding_dim=embedding_dim,
                    rnn_units=rnn_units,
                    batch_size=BATCH_SIZE)


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                           logits,
                                                           from_logits=True)


model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, save_weights_only=True)

EPOCHS = 10
history = model.fit(text_ds, epochs=EPOCHS, callbacks=[checkpoint_callback])
