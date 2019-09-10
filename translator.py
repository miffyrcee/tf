import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class attention(tf.keras.models.Model):
    def __init__(self, units):
        super().__init__()
        self.w1 = tf.keras.layers.Dense(units)
        self.w2 = tf.keras.layers.Dense(units)
        self.v = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_time_axis = tf.expand_dims(query, axis=1)
        scores = self.v(self.w1(values) + self.w2(hidden_time_axis))
        weights = tf.nn.softmax(scores, axis=1)
        context_vector = weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return weights, context_vector


class encoder():
    def __init__(self, vocab_size, embedding_dim, units):
        super().__init__()
        self.em = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units)

    def call(self, x, hidden):
        x = self.em(x)
        output, state = self.gru(x, initial_state=hidden)
