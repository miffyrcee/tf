import sys
from itertools import product
from random import choices, shuffle

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfh
import tensorflow_probability as tfp

actions = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])  #(left,down,up,right)

mcmc_map = np.arange(16.).reshape(4, 4)
mcmc_map[3][1] = -np.inf
print(mcmc_map)
rewards = np.random.uniform((3, 4, 4))
gamma = 0.2
distance = 1
gamma = gamma**distance


def walk_random(path, current_state_num, depth=0):
    if current_state_num not in path:
        path[depth] = current_state_num
    for action in actions:
        try:
            current_state_pos = tf.unravel_index(current_state_num, (4, 4))
            next_state = current_state_pos + action
            next_state_num = mcmc_map[int(next_state[0])][int(next_state[1])]
            if next_state_num == -np.inf or next_state[0] < 0 or next_state[
                    1] < 0:  # print('trap')
                continue
            elif next_state_num == 15:
                sys.exit(0)

            else:
                if next_state_num not in path:
                    walk_random(path, next_state_num, depth=depth + 1)

        except IndexError:
            continue
            print('side')
    print(path)


walk_random([None] * 18, 11)
