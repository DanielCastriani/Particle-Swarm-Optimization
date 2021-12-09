import numpy as np
import plotly.graph_objects as go

from py_pso import PSO
from py_pso.dtypes import Bound
from utils.plot_utils import draw_optimization_surface, plot_surface


def objective_function(vector: np.ndarray):
    return sum(vector ** 2)


c1 = .07
c2 = .08
w = .05

n_iter = 100
n_particles = 30

Bounds: Bound = [[-100, 100], [-100, 100]]

optimizer = PSO(objective_function=objective_function, n_particles=n_particles, bound=Bounds, c1=c1, c2=c2, w=w)

best, score, hist = optimizer.optimize(iteration=n_iter, seed=1234, return_hist=True)

plot_surface(objective_function, bounds=Bounds, best_position=best)


frames = [draw_optimization_surface(objective_function, bounds=Bounds, particles=h) for h in hist]

fig = go.Figure(
    data=frames[0],
    layout=go.Layout(
        title="PSO", hovermode="closest",
        updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]),
    frames=[go.Frame(data=f) for f in frames[1:]])

fig.show()
