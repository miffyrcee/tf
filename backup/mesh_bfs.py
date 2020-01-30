from collections import defaultdict, deque
from random import choices

import gym
import numpy as np
import tensorflow as tf


class node(object):
    def __init__(self, v, p):
        self.p = p
        self.v = v


class mesh_search(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.mesh_map = np.arange(16.).reshape((4, 4))
        self.mesh_map[0][0] = 0
        self.mesh_map[3][3] = 0
        self.sx, self.sy = self.mesh_map.shape
        self.probability = defaultdict(node)
        self.actions = [[-1, 0], [0, -1], [0, 1], [1, 0]]

    def greed_search(self, start, end):
        visited = defaultdict(node)
        visited[start] = node(start, f=0, d=0)
        heap = [visited[start]]
        while heap:
            ele = heap.pop(0)
            if ele.v == end:
                return self._all(ele)
                break
            for action in self.actions:
                x, y = tf.unravel_index(ele.v, (self.sx, self.sy)) + action
                if 0 < x < self.sx and 0 < y < self.sy:
                    n = np.matmul([x, y], [self.sy, 1])
                    if self.mcmc_map[x][y] != -np.inf and visited[n].d == -1:
                        visited[n] = node(n, f=ele.f + 1, p=ele, d=0)
                        heap.append(visited[n])
                        heap
            ele.d = 1

    def pure_mdp(self, gamma):
        # st = (trans_prob,next_state,rewards,label)

        pass

    def greed_policy(self):
        print('\a')

        pass

    def value_iter(self):
        pass

    def _all(self, ele):
        stack = deque()
        while ele is not None:
            stack.appendleft(ele.v)
            ele = ele.p
        return stack


def mesh_ele(object):
    def __init__(self):
        pass


class mesh_env(object):
    def __init__(self):
        self.mesh_render = np.arange(48.).reshape((4, 12))

    def states(self):
        state = defaultdict(node())


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    print(np.arange(env.nS).reshape(4, 12))
    print(env.P)
