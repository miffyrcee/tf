import os
import socket

import matplotlib.pyplot as plt
import numpy as np
# for _ in range(1000):
#     for x, y in dataset:
#         # train_step(x, y)
#         print(model(x))
import numpy.polynomial as poly
import tensorflow as tf
import tensorflow_probability.python as tfp
from numpy.polynomial import Polynomial as P

tfd = tfp.distributions

port_scanner = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    port_scanner.connect(('127.0.0.1', 8089))
    import socks
    socks.set_default_proxy(socks.SOCKS5, '127.0.0.1', 8089)
    socket.socket = socks.socksocket
except ConnectionRefusedError:
    print('no port')

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(train_dataset_url), origin=train_dataset_url)


def load_data(fp):
    dataset = tf.data.TextLineDataset(fp)
    dataset = dataset.skip(1)

    def in_out(line):
        return tf.py_function(split, inp=[line], Tout=[tf.float32] * 2)

    def split(x):
        x = x.numpy().decode('utf-8').split(',')
        x = tf.strings.to_number(x)
        return x[:-1], x[-1]

    dataset = dataset.map(in_out)
    dataset = dataset.filter(lambda x, y: y < 2)
    dataset = dataset.shuffle(20)
    dataset = dataset.batch(20)

    return dataset


dataset = load_data(train_dataset_fp)


class svm(tf.keras.Model):
    def __init__(self, input_features, output_features, name=None):
        super().__init__(name=name)
        p = poly.polypow(input_features * [1], 2)

        self.d1 = tf.keras.layers.Dense(
            input_features,
            input_shape=(input_features, ),
        )

    def kernel_func(self):
        self.d2 = tf.keras.layers.Dense(4 * 4, activation='relu')

    def call(self, x):
        y = self.d1(x)
        return y


optimizer = tf.optimizers.Adam()
# model = svm(4, 4)


@tf.function()
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model.call(x)
        loss = tf.losses.hinge(y, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


x, y = next(iter(dataset))


def model(x):
    feature = x.shape[-1]
    # p = poly.polymul(feature * [1], feature * [1])**0.5

    # p = poly([1,2,3])
    p = poly.Polynomial(coef=[1, 1, 1])
    print(p.domain)
    print(p.window)
    # q = poly.Hermite(coef=[1, 1])
    print(p * p)
    # w = np.random.normal(size=(feature, len(p)))

    return p


r = model(x[:, :3])
print(r)
