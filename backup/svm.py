import numpy as np

import tensorflow as tf
import tensorflow_probability.python as tfp

p = np.array([[1, 2], [3, 4]])

print(p @ p.T)
a = np.eye(2) * 2
print(p @ a @ p.T)

x = tf.linspace(-1., 1, num=1000)
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = x**3 - 2 * x**2 + 2
    grad = tape.gradient(y, [x])
hessian = tape.gradient(grad[0], [x])
img_raw = tf.io.read_file('/home/miffyrcee/Pictures/shilpasraomagicpixo1_dewdrops1.jpg')
png = tf.io.decode_jpeg(img_raw)
print(tf.keras.layers.Flatten()(png))
print(png)
