import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def normal(x, mu=0, sigma=1):
    p = 1 / np.ma.sqrt(2 * np.pi * sigma**2) * np.ma.exp(
        -(x - mu)**2 / 2 * sigma**2)
    return p


def kernel_func(x1, x2):
    return normal(x1 + x2)


class node(object):
    def __init__(self, x, y, p=None):
        self.x = x
        self.y = y
        self.p = p


class k_d_tree(object):
    def __init__(self):
        self.obj0 = np.sort(np.random.uniform(size=(10, 2)), axis=0)
        self.obj1 = np.sort(np.random.uniform(size=(10, 2)), axis=0)
