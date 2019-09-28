import numpy as np
import tensorflow as tf

em1 = tf.keras.layers.Embedding(100, 64)(tf.range(100))


class gru(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.f = tf.keras.layers.Dense(64)
        self.i = tf.keras.layers.Dense(64)
        self.o = tf.keras.layers.Dense(64)
    def call(self):
