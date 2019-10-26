import gc
import heapq
import math
import sys
from collections import defaultdict, deque
from copy import copy

import numpy as np
import tensorflow as tf

mcmc_map = np.arange(30.).reshape(5, 6)
mcmc_map[3][1] = -np.inf
mcmc_map[2][1] = -np.inf
#(left,down,up,right)


class node(object):
    def __init__(self, v=-1, f=-1, d=-1, p=None):
        self.v = v
        self.f = f  #find
        self.d = d  #finish
        self.p = p


class walk_random():
    def __init__(self):
        self.sx, self.sy = mcmc_map.shape
        self.actions = [[-1, 0], [0, -1], [0, 1], [1, 0]]
        self.mcmc_map = mcmc_map

    def bfs(self, start, end):
        # best vector
        # f可以不写,此刻可能用来推测mdp的概率
        visited = defaultdict(node)
        visited[start] = node(start, f=0, d=0)
        heap = [visited[start]]
        while heap:
            ele = heap.pop(0)
            if ele.v == end:
                self._all(ele)
                break
            for action in self.actions:
                x, y = tf.unravel_index(ele.v, (self.sx, self.sy)) + action
                n = np.matmul([x, y], [self.sy, 1])
                if self.boundary(
                        x, y
                ) and self.mcmc_map[x][y] != -np.inf and visited[n].d == -1:
                    visited[n] = node(n, f=ele.f + 1, p=ele, d=0)
                    heap.append(visited[n])
            ele.d = 1

    def min_depth(self, d):
        if self.state.d < d:
            return 1
        pass

    def boundary(self, x, y):
        if 0 <= x < self.sx and 0 <= y < self.sy:
            return 1

    def insert(self):

        pass

    def max_f(self, visited):
        return max([v.f for v in visited.values()])

    def _all(self, ele):
        stack = deque()
        while ele is not None:
            stack.appendleft(ele.v)
            ele = ele.p
        print(stack)


t = walk_random()
print(mcmc_map)
t.bfs(1, 29)
q = defaultdict(node)
