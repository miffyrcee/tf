import numpy as np
import scipy.spatial as spatial

test = np.random.uniform(size=(10, 2))
point = np.random.uniform(size=(4, 2))
stree = spatial.KDTree(point)
_, query_result = stree.query(test, k=4)
print(query_result)
