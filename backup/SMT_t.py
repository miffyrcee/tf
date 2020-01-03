import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

units = 64


class att_block(tf.keras.layers.Layer):
    '''
    quoted from https://arxiv.org/pdf/1903.03878.pdf
    '''
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.dx = x.shape[-1]
        self.dy = y.shape[-1]
        self.wq = tf.keras.layers.Dense(self.dx)
        self.wk = tf.keras.layers.Dense(self.dy)
        self.wv = tf.keras.layers.Dense(self.dy)
        self.d = tf.keras.layers.Dense(self.dy)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.att = tf.keras.layers.Attention(use_scale=False)

    def call(self, u, v, k):
        self.xwu = self.wq(u)
        self.ywk = self.wk(k)
        self.ywv = self.wv(v)
        self.num_heads = self.ln1(
            self.att(self.xwu, self.ywk, self.ywv) + self.x)
        self.att_block = self.ln2(self.d(self.num_heads) + self.num_heads)
        return self.att_block


class mdp(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()


class encoder(tf.keras.models.Model):
    '''
    states
    actions
    transitions
    rewards
    '''
    def __init__(self):
        super().__init__()
        self.p = tf.random((units, units))
        self.p = tf.keras.layers.Dense(units)(self.p)
        self.em = tf.keras.layers.Embedding()
        self.att_b = att_block()

    def call(self, u, v, k):
        self.em()
