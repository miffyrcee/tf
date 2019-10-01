import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

units = 64


def att(u, k, v):
    uk = tf.matmul(u, k, transpose_b=True)
    att_weights = tf.nn.softmax(uk)
    return tf.matmul(att_weights, v)


def probabilities(x, y, theta, scaled_factor, t):
    return x / scaled_factor, y / scaled_factor, tf.sin(theta), tf.cos(
        theta), tf.math.exp(-t)


class att_block(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.fc = tf.keras.layers.Dense(units)
        self.att_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.h_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.wu = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

    def call(self, x, y):
        xwu = tf.matmul(x, self.wu)
        ywk = tf.matmul(y, self.wk)
        ywv = tf.matmul(y, self.wv)
        hidden = self.h_ln(att(xwu, ywk, ywk), x)
        att_block = self.att_ln(self.fc(hidden) + hidden)
        return self.att_block


class encoder_layer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()


class encoder(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, mask):
        return att_block(mask, mask)
