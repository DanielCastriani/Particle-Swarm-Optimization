import numpy as np

from py_pso import PSO
from py_pso.dtypes import Bound


def objective_function(vector: np.ndarray):
    return sum(vector ** 2)


c1 = .7
c2 = .8
w = .5

n_iter = 100
n_particles = 30

bound: Bound = [[-10, 10], [-10, 10]]

optimizer = PSO(objective_function=objective_function, n_particles=n_particles, bound=bound, c1=c1, c2=c2, w=w)

best, score = optimizer.optimize(iteration=n_iter, seed=1234)
