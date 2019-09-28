import numpy as np


def get_next_table(P):
    i = 0
    j = -1
    next_table = [-1]

    while i < len(P) - 1:
        if j == -1 or P[i] == P[j]:
            i += 1
            j += 1
            next_table.insert(i, j)
        else:
            j = next_table[j]

    return next_table


def index_kmp(S, P, pos):
    next_table = get_next_table(P)
    i = pos
    j = 0

    while i <= len(S) - 1 and j <= len(P) - 1:
        if j == -1 or S[i] == P[j]:
            i += 1
            j += 1
        else:
            j = next_table[j]

    if j == len(P):
        return i - j
    else:
        return -1
