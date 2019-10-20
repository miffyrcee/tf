from random import choices

import numpy as np

# finite automata
p = [0, 1, 0, 1, 0, 2, 0]
p = np.array(p)

gamma = np.zeros((8, 3))

for i, j in enumerate(p):
    gamma[i][j] = i + 1
for i in np.where(p == 0)[0]:
    gamma[i + 1][p[i]] = 1
gamma[5][1] = 4
gamma[7][1] = 2
print(gamma)
test = choices(range(3), k=10000)

phi = []  #final func
t = 0
for i, j in enumerate(test):
    t = int(gamma[t][j])
    if t == 7:
        phi.append(i)
print(phi)
