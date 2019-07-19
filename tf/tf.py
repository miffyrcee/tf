# -*- coding: utf-8 -*-
"""Main module."""
import scipy
import tensorflow as tf

c = (scipy.sum(map(lambda x: pow(x, 2), scipy.fft(range(10)))))
print([i for i in c])
