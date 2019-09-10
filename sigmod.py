import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

x = np.linspace(-4, 4)
y = np.ma.tanh(x)
plt.plot(x, y)
plt.plot(0, np.ma.tanh(0), 'o')
plt.show()
