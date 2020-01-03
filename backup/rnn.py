# Plot the PDF.
import functools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability.python as tfp

plt.style.use("ggplot")


class rnn():
    def __init__(self, hidden):
        self.hidden = hidden
        self.t = 0
        self.w = list()
        self.u = list()
        self.v = list()

    def update(self):
        pass

    def __call__(self, x, b, c):
        self.hidden[self.t] = self.hidden[self.t] * self.w + self.u * x + b
        self.hidden[self.t + 1] = tf.tanh(self.hidden)
        self.output = self.hidden[self.t + 1] * self.v + c
        self.output = tf.math.softmax(self.output)


tfd = tfp.distributions


def model():
    tfd.Normal(loc=1, scale=10, name='avg_affect')


# tfd.JointDistributionSequential(model)
x = tf.linspace(-2., 2., 1000)
y = tfd.Cauchy(1, 0.5).prob(x)
