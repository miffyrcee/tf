import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability.python as tfp

tfd = tfp.distributions

p = tfp.distributions.Normal(loc=1., scale=1.)
q = tfp.distributions.Normal(loc=0., scale=2.)

# loss_p = tf.nn.sigmoid_cross_entropy_with_logits(
#     labels=tf.zeros_like(sample_p), logits=sample_p)

sample_p = p.sample((5000, 1))
sample_q = q.sample((5000, 1))

models = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_dim=1, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])


def bin_loss(sample_p, sample_q):
    loss_p = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(sample_p), logits=models(sample_p))
    loss_q = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(sample_q), logits=models(sample_q))
    return tf.reduce_mean(loss_q + loss_q)
