import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

units = 64


def att(u, k, v):
    uk = tf.matmul(u, k, transpose_b=True)
    att_weights = tf.nn.softmax(uk)
    return tf.matmul(att_weights, v)


class att_block(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.fc = tf.keras.layers.Dense(units)
        self.att_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.h_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.wu = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

    def call(self, x, y):
        self.xwu = tf.matmul(x, self.wu)
        self.ywk = tf.matmul(y, self.wk)
        self.ywv = tf.matmul(y, self.wv)
        self.hidden = self.h_ln(att(self.xwu, self.ywk, self.ywv) + x)
        self.att_block = self.att_ln(self.fc(self.hidden) + self.hidden)
        return self.att_block


class encoder(tf.keras.models.Model):
    def __init__(self):
        super().__init__()

    def call(self, mask):
        return att_block(mask, mask)
