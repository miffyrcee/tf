import functools

import numpy as np
import sympy
from sympy.abc import a, b, x, y

system = sympy.Matrix([[9 * x, 3], [10 * y, 6]])
print(sympy.solve_linear_system(system, x, y))

