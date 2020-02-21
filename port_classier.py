import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability.python as tfp

tfd = tfp.distributions

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(4, )),
    tf.keras.layers.Conv3D(),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])
