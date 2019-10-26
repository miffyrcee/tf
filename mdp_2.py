import sys
from collections import defaultdict, deque, namedtuple

import numpy as np
import tensorflow as tf

mcmc_map = np.arange(16.).reshape(4, 4)
mcmc_map[3][1] = -np.inf
print(mcmc_map)
actions = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])  #(left,down,up,right)
x_max, y_max = mcmc_map.shape

path = namedtuple('path', ['p', 'v'])


def print_path(pa):
    if pa.p is not None:
        yield from print_path(pa.p)
        return pa.p


def walk(state, node, depth=0):
    paths = [ne for ne in print_path(node)]
    if depth == 7:
        sys.exit(0)
    for action in actions:
        next_state_pos = tf.unravel_index(state, (x_max, y_max)) - action
        x = next_state_pos[0]
        y = next_state_pos[1]
        if 0 <= x < x_max and 0 <= y < y_max:
            next_state_num = mcmc_map[x][y]
            print(next_state_num)
            if next_state_num == -np.inf:
                continue
            elif state == 15:
                print(paths)
                return 0
            else:
                if next_state_num not in paths:
                    walk(next_state_num, path(node, v=state), depth + 1)

walk(0, node=path(None, v=0))
