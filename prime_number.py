import matplotlib.pyplot as plt
import numpy as np

n = 1000
theat = 0.5
r = 1
for i in range(n):
    x = r * np.cos(theat * i) * i
    y = r * np.sin(theat * i) * i
    plt.plot(x, y, 'o')
plt.savefig('c.png')
plt.show()

print(np.pi)
