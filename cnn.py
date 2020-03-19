import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3),
                           activation=tf.nn.relu,
                           input_shape=(100, 50, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])

path = pathlib.Path('image')
label = np.array([os.path.basename(it) for it in path.glob('*')])


def load_data():
    dataset = tf.data.Dataset.list_files('image/**/*.jpeg')
    path = pathlib.Path('image')
    label = np.array([os.path.basename(it) for it in path.glob('*')])

    def decode_img(img):
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, size=[100, 50])

    def get_label(img):
        line = tf.strings.split(img, '/')
        return label == line[-2]

    def split(img):
        return decode_img(img), get_label(img)

    dataset = dataset.map(split, num_parallel_calls=True)
    return dataset


train_ds = load_data().batch(4, drop_remainder=True)
optimizer = tf.optimizers.Adam(learning_rate=0.003)
optimizer.


@tf.function()
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model.call(x)
        loss = tf.losses.hinge(y, y_pred)
        # loss = tf.reduce_sum(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    tf.print('loss:', tf.reduce_mean(loss))
    # tf.print('pred:', y_pred)
    return loss


for _ in range(50):
    for x, y in train_ds.take(16):
        train_step(x, y)

imgs = path.glob('**/*.jpeg')


def evaluate():
    fig, axs = plt.subplots(4, 4)

    for i, y in enumerate(load_data().shuffle(4).take(16)):
        _x, _y = tf.unravel_index(i, (4, 4))
        axs[_x, _y].imshow(y[0])
        lab = label[model.call(y[0][tf.newaxis, :]) > 0]
        axs[_x, _y].set_title(lab[0])
        # axs.title(lab)

    plt.subplot_tool()
    plt.show()


evaluate()



