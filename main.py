from __future__ import absolute_import, division, print_function

import glob
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_estimator
import tensorflow_hub as hub
from IPython.display import Image, display
from tensorflow import keras

(train_images,
 train_labels), (test_images,
                 test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784, )),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


def adjust(image):
    tf.image.adjust_hue(image)
    tf.image.adjust_saturation(image)
    return image


model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('my_model.h5')

arr = np.zeros((2, 2), int)
print(arr)
