import itertools
from random import random

import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('Blackjack-v0')
policy = np.random.uniform(size=(env.action_space.n))
policy = policy / np.sum(policy)


def v_update():
    state = env.reset()
    print('玩家 = {}, 庄家 = {}'.format(env.player, env.dealer))
    print('state:', state)
    for _ in range(10):

        action = np.random.choice(range(env.action_space.n), p=policy)
        print('action:', action)
        state, reward, done, _ = env.step(action)
        print(state, reward, done)
        if done:
            break


for _ in range(10):
    v_update()
policy = np.zeros((22, 11, 2, 2))
policy[20:, :, :, 0] = 1  # >= 20 时收手
policy[:20, :, :, 1] = 1
