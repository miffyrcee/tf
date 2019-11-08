import math

import gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice

np.random.seed(0)

env = gym.make('Taxi-v3')


def v_update(gamma=0.95):
    policy = np.random.uniform(size=(env.nA))
    policy = policy / np.sum(policy)
    v = np.zeros(env.nS)
    for _ in range(100):
        for state in range(env.nS):
            action = np.random.choice(range(env.nA), p=policy)
            v_s = 0
            for prob, next_state, reward, done in env.P[state][action]:
                v_s += prob * (reward + gamma * v[next_state])
            v[state] = max(v_s, v[state])
    return v


v = v_update()


def pi_update(v, gamma=0.95):
    policy = np.random.uniform(size=(env.nA))
    policy = policy / np.sum(policy)
    pi = np.zeros((env.nS, env.nA))
    for _ in range(100):
        for state in range(env.nS):
            action = np.random.choice(range(env.nA), p=policy)
            for prob, next_state, reward, done in env.P[state][action]:
                pi_sa = prob * (reward + gamma * v[next_state])
                pi[state][action] = max(pi[state][action], pi_sa)
    return np.argmax(pi, axis=1)


pi = pi_update(v)
print(np.eye(env.nA)[pi])
