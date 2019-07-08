import re
import shlex
from collections.abc import Iterable

import tensorflow as tf

path = [[32, 6], [35], 35, [36, 662, 36, [345]],
        [23, 26, [235, [26], [26, [61]]]]]


def flatten(arr):
    stack = list()
    arr = iter(arr)
    while 1:
        try:
            i = next(arr)
        except StopIteration:
            if stack:
                arr = stack.pop(0)
                continue
            else:
                return
        if isinstance(i, Iterable):
            stack.append(arr)
            arr = iter(i)
        else:
            yield i
