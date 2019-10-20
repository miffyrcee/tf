import sys
from itertools import product
from random import choices, shuffle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfh
import tensorflow_probability as tfp

map_t = [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]]


def neigh_pos(pos):
    x, y = pos
    return [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]


p = [[[]]]
s = [0, 1, 2, 3]
a = [0, 1, 2, 3]


def path(state, pos):
    next_s = max(p[state])
    p[s][a][next_s]


def forward(x, y):
    for i, j in [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]:
        if i % 4 and j % 3:
            forward(i, j)
        else:
            sys.exit(0)
