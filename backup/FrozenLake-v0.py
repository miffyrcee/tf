import heapq

import gym
import numpy as np
from numpy.random import choice

env = gym.make('FrozenLake-v0')

random_policy = np.random.uniform(size=(env.nS, env.nA))
random_policy = random_policy / np.sum(random_policy, axis=1)[:, np.newaxis]
# random_policy = np.ones((env.nS, env.nA)) * 0.25


def v_update(policy, gamma):
    v = np.zeros((env.nS))
    for _ in range(100):
        for state in range(env.nS):
            v_state = list()
            action = choice(range(env.nA), p=policy[state])
            for prob, next_state, reward, done in env.P[state][action]:
                v_state.append(prob * (reward + gamma * v[next_state]))
            v[state] = max(v[state], sum(v_state))

    return v


v = v_update(random_policy, gamma=1.)


def q_update(v, policy, gamma):
    q = np.zeros((env.nS, env.nA))
    for _ in range(100):
        for state in range(env.nS):
            action = choice(range(env.nA), p=policy[state])
            q_sa = list()
            for prob, next_state, reward, done in env.P[state][action]:
                q_sa.append(prob * (reward + gamma * v[next_state]))

            q[state][action] = max(q[state][action], sum(q_sa))
    return q


q = (q_update(v, random_policy, gamma=1.))
print(q)
print(np.argmax(q, axis=1))
