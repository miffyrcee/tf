from collections.abc import Iterable
from random import choices

import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy

env = gym.make('CliffWalking-v0')

gamma = 0.1
policy = np.random.uniform(size=(env.nS, env.nA))
policy = policy / np.sum(policy, axis=1)[:, np.newaxis]

a, b = np.eye(env.nS), np.zeros((env.nS))
for state in range(env.nS - 1):
    for action in range(env.nA):
        pi = policy[state][action]
        for p, next_state, reward, done in env.P[state][action]:
            a[state, next_state] -= (pi * gamma * p)
            b[state] += (pi * reward * p)
v = np.linalg.solve(a, b)
print(np.argsort(v))
x0 = [0, 1, 2]
x = [0.2, 0.3, 0.5]
y = np.exp(x) / sum(np.exp(x))
plt.plot(x0, x)
plt.plot(x0, y, 'o')
