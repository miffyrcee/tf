import numpy as np

strings = 'asgasdgasgdashhadsdd'
a = np.array(list(strings))

current_char = 's'
pos = np.argwhere(a == current_char)
print(pos[..., -1])
current_char = strings[-1]
pos = np.argwhere(a == current_char)
print(pos[..., -1])
current_char = strings[-2]
pos = np.argwhere(a == current_char)
print(pos[..., -1])
