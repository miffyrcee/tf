import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds


class att_block(tf.keras.layers.Layer):
    '''
    quoted from https://arxiv.org/pdf/1903.03878.pdf
    '''
    def __init__(self, dx, dy):
        super().__init__()
        self.wq = tf.keras.layers.Dense(dx)
        self.wk = tf.keras.layers.Dense(dy)
        self.wv = tf.keras.layers.Dense(dy)
        self.d = tf.keras.layers.Dense(dy)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.att = tf.keras.layers.Attention(use_scale=False)

    def call(self, x, y):
        self.xwu = self.wq(x)
        self.ywk = self.wk(y)
        self.ywv = self.wv(y)
        self.num_heads = self.ln1(self.att(self.xwu, self.ywk, self.ywv) + x)
        self.att_block = self.ln2(self.d(self.num_heads) + self.num_heads)
        return self.att_block
