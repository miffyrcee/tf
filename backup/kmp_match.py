from random import choices

import numpy as np

# finite automata
# O(n)*2
p = [0, 1, 0, 1, 0, 2, 0]
p = np.array(p)

# create trans_table
sigma = np.zeros((8, 3))

for i, j in enumerate(p):
    sigma[i][j] = i + 1
for i in np.where(p == 0)[0]:
    sigma[i + 1][p[i]] = 1
sigma[5][1] = 4
sigma[7][1] = 2
print(sigma)
test = choices(range(3), k=10000)

phi = []  #final func
t = 0
for i, j in enumerate(test):
    t = int(sigma[t][j])
    if t == 7:
        phi.append(i)
print(phi)
