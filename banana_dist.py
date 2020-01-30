import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability.python as tfp

tfd = tfp.distributions
sigma = [[1., 0.95], [0.95, 1]]
p_x = tfd.MultivariateNormalTriL(scale_tril=tf.linalg.cholesky(sigma))

exp = tfp.bijectors.Inline(
    forward_fn=tf.exp,
    inverse_fn=tf.math.log,
    inverse_log_det_jacobian_fn=lambda y: -tf.math.log(y),
    forward_min_event_ndims=0,
    name='exp')
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4))
u1_lim = -3., 3.  # -2.5, 2.5
u2_lim = -10., 2.  # -7.0, 2.0


class Banana(tfp.bijectors.Bijector):
    def __init__(self, name="banana"):
        super(Banana, self).__init__(
            inverse_min_event_ndims=1,
            #                                      is_constant_jacobian=True,
            name=name)

    def _forward(self, x):

        y_0 = x[..., 0:1]
        y_1 = x[..., 1:2] - y_0**2 - 1
        y_tail = x[..., 2:-1]

        return tf.concat([y_0, y_1, y_tail], axis=-1)

    def _inverse(self, y):

        x_0 = y[..., 0:1]
        x_1 = y[..., 1:2] + x_0**2 + 1
        x_tail = y[..., 2:-1]

        return tf.concat([x_0, x_1, x_tail], axis=-1)

    def _inverse_log_det_jacobian(self, y):

        return tf.constant(0.)


#         return tf.zeros(shape=y.shape[:-1])

banana = Banana()
p_y = tfd.TransformedDistribution(distribution=p_x, bijector=banana)
ax1.plot(*p_y.sample((1000)).numpy().T, '.', alpha=0.6)
plt.show()
