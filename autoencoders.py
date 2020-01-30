import socket

import matplotlib.pyplot as plt
import numpy as np
import socks
import tensorflow as tf
import tensorflow_probability.python as tfp

socks.set_default_proxy(socks.SOCKS5, '127.0.0.1', 8089)
socket.socket = socks.socksocket

tfd = tfp.distributions

data = [[1.2, 1.0, 3.2, 1.2], [1.2, 1.5, 4.9, 1.14], [1.3, 1.0, 6.3, 2.2]]
data = tf.constant(data)

print(tf.keras.layers.Embedding(input_dim=16, output_dim=1)(data))


class kl_vi_layer(tf.keras.layers.Layer):
    def __init__(self, p, q):
        super().__init__(self)
    def call(self):
        pass
