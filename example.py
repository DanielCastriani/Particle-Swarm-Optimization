import matplotlib.pyplot as plt
import numpy as np
from py_pso import PSO
from py_pso.dtypes import Bound
from utils.functions import bird_function, townsend_function, townsend_function_bounds
from utils.plot_utils import plot_animation, plot_surface


def objective_function(vector: np.ndarray):
    return bird_function(vector[0], vector[1])
    

bounds: Bound = townsend_function_bounds

plot_surface(objective_function, bounds)


c1 = .09
c2 = .12
w = .06

n_iter = 1000
n_particles = 30

optimizer = PSO(objective_function=objective_function, n_particles=n_particles, bound=bounds, c1=c1, c2=c2, w=w)

best, score, hist = optimizer.optimize(iteration=n_iter, seed=1234, return_hist=True)

best, score

plot_animation(hist, bounds=bounds, objective_function=objective_function, save_path='img', filename='pso_townsend.gif')
plt.show()




