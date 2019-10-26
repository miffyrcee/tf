import sys
import time

pre_time = time.time()


def test(path, count=0):
    if count < len(path):
        for i in ['up', 'down', 'left', 'right']:
            path[count] = i
            test(path, count + 1)
        return 0
    else:
        print(path)
        return 0

test(path=[3, 3, 3])
