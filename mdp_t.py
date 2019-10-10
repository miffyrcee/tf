import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

'''
states
actions
transitions
rewards
prob = pr
'''

# q_state
# v_state
# pi_state
pos = tf.ones((3, 4))
poss = tf.ones((3, 4))

x, y = (1, 2)


def explorer(u, n, k=0):
    return u + k / n
