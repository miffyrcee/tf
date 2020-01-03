import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    scaled_matual_qk = matmul_qk / tf.math.sqrt(
        tf.cast(matmul_qk.shape[-1], tf.int32))
    if mask is not None:
        scaled_matual_qk += (mask * -1e9)
    attention_weights = tf.nn.sigmoid(scaled_matual_qk)
    output = tf.matmul(attention_weights, v)


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff,
                              activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class multi_head_attention(tf.keras.layers.Layer):
    def __init__(self):
        pass


class encoder_layer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.ln = tf.keras.layers.LayerNormalization(epsilon=0.001)
