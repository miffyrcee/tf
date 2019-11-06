import itertools
from random import random

import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('Blackjack-v0')
policy = np.random.uniform(size=(env.action_space.n))
policy = policy / np.sum(policy)

for _ in range(10):
    action = np.random.choice(range(env.action_space.n), p=policy)
    observation, reward, done, _ = env.step(action)
    print(observation, done)
    if done:
        break
