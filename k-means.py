from time import sleep

import matplotlib.pyplot as plt
import numpy as np

k = 3
p0 = np.random.uniform(high=0.5, size=(2, 10050))
p1 = np.random.uniform(low=0.5, size=(2, 10050))
p = np.hstack((p0, p1))


def m_clclator(m):
    d = np.sum((p - m[:, 0][:, np.newaxis])**2, axis=0)[:, np.newaxis]
    for i in range(1, k):
        l2 = np.sum((p - m[:, i][:, np.newaxis])**2, axis=0)[:, np.newaxis]
        d = np.hstack((d, l2))
    labels = np.argmin(d, axis=-1)
    for i in range(k):
        loc = np.argwhere(labels == i).flatten()
        m[:, i] = np.sum(p[:, loc], -1) / len(loc)
    plt.plot(m[0], m[1], 'o')
    return m


m = np.random.uniform(size=(2, k))
for _ in range(10):
    s = np.var(m)
    m = m_clclator(m)
    print(s)
