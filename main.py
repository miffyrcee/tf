import os
import random as r
import re
import sys

import tensorflow as tf
import tensorflow_hub as th
from tensorflow import io

path = '/home/miffyrcee/Downloads'
gif_name = [
    os.path.join(path, i) for i in os.listdir(path)
    if re.match('(\w)+.jpeg', i)
]
(train_images,
 train_labels), (test_images,
                 test_labels) = tf.keras.datasets.mnist.load_data()

os.path()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


class mnist_model(tf.keras.Model):
    def __init__(self):
        super(mnist_model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = mnist_model(images)
        loss = tf.losses.SparseCategoricalCrossentropy(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.3)
if '__name__' == '__main__':
    pass
