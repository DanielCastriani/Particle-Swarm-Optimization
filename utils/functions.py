
import math

import numpy as np
from py_pso.dtypes import Bound


def bird_function(x: float, y: float):
    return np.sin(x)*(np.exp(1-np.cos(y))**2)+np.cos(y)*(np.exp(1-np.sin(x))**2)+(x-y)**2


bird_function_bounds: Bound = [[-2*math.pi, 2*math.pi], [-2*math.pi, 2*math.pi]]


def townsend_function(x: float, y: float):
    return (np.cos(x-0.1) * y) ** 2 - x * np.sin(3*x + y)


townsend_function_bounds: Bound = [[-2.25, 2.25], [-2.5, 1.75]]
