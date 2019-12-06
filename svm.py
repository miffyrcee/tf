import numpy as np
import tensorflow as tf
import tensorflow_probability.python as tfp

p = np.array([[1, 2], [3, 4]])

print(p @ p.T)
a = np.eye(2) * 2
print(p @ a @ p.T)

x = tf.linspace(-1., 1, num=1000)
with tf.GradientTape(persistent=True) as tg:
    tg.watch(x)
    y = x**3 - 2 * x**2 + 2
    grad = tg.gradient(y, [x])
hessian = tg.gradient(grad[0], [x])
