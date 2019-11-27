from itertools import permutations, product

import matplotlib.pyplot as plt
import numpy as np

p = np.random.uniform(size=(2, 10000))
p_sorted = np.argsort(p, -1)
p_x = p_sorted[0]
p_y = p_sorted[1]


def factor(loc):  # -> tuple
    mid = len(loc) // 2  # -> int
    left = loc[:mid]  # -> list
    right = loc[mid:]
    return left, right, loc[mid]


def solve(loc):
    min_d = np.inf
    if len(loc) == 1:
        return min_d
    for i, j in permutations(loc, 2):
        min_d = min(min_d, sum(p[:, i] - p[:, j])**2)
    return min_d


def merge(left, right, mid, d):
    min_d = d
    xls = list()  # ¦d¦ ¦
    for i in left[:-1]:
        if abs(p[0][i] - p[0][mid]) <= d**2:
            xls.append(p_x[i])
        else:
            break
    xrs = list()  # ¦ ¦d¦
    for i in right:
        if abs(p[0][i] - p[0][mid]) <= d**2:
            xrs.append(p_x[i])
        else:
            break
    for i, j in product(xls, xrs):  # need improved
        if i != j:
            if abs(p[1][i] - p[1][j]) <= 4 * d**2:
                min_d = min(min_d, np.sum((p[:, i] - p[:, j])**2))
    return min_d


def min_dd(loc):
    left, right, mid = factor(loc)
    if len(left) <= 3:
        delta_left = solve(left)
    else:
        delta_left = min_dd(left)
    if len(left) <= 3:
        delta_right = solve(right)
    else:
        delta_right = min_dd(right)
    delta = min(delta_left, delta_right)
    return merge(left, right, mid, delta)


def main():
    print(np.ma.sqrt(min_dd(p_x)))


if __name__ == "__main__":
    main()
