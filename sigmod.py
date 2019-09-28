import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

x = np.linspace(0, 1)
y = np.ma.log2(x)

plt.plot(x, y)
plt.show()
