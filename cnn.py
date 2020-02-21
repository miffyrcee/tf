import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3),
                           activation=tf.nn.relu,
                           input_shape=(64, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])


def load_data():
    dataset = tf.data.Dataset.list_files('image/**/*.jpeg')

    def processor(img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, size=(50, 50))

    def split(img):
        print(img)
        line = tf.strings.split(img, '/')
        return processor(img), line[-2]

    return dataset.map(split)


train_ds = load_data()
for i in train_ds.take(1):
    print(i)
