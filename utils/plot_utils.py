from typing import cast

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
from py_pso.dtypes import Bound, ObjectiveFunction
from py_pso.particle import Particle

from utils.file_utils import create_path


def _get_surface(bounds: Bound = None, x: NDArray[np.float_] = None, y: NDArray[np.float_] = None):
    if bounds is None and x is None and y is None:
        raise ValueError("You should define 'bounds' or 'x' and 'y'")

    if x is None and bounds is not None:
        x = np.arange(*bounds[0])

    if y is None and bounds is not None:
        y = np.arange(*bounds[1])

    x, y = np.meshgrid(x, y)

    x, y = cast(NDArray[np.float_], x), cast(NDArray[np.float_], y)

    return x, y


def plot_animation(
        hist: list[Particle],
        objective_function: ObjectiveFunction,
        bounds: Bound = None,
        x: NDArray[np.float_] = None, y: NDArray[np.float_] = None,
        figsize: tuple[int, int] = (10, 10), save_path: str = None, filename: str = None):

    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['animation.embed_limit'] = 16384
    plt.rcParams['figure.dpi'] = 150
    plt.ioff()

    def plot_frame(i: int, hist: list, ax):
        ax.clear()

        particles = hist[i]

        X, Y = _get_surface(bounds, x=x, y=y)
        Z = np.array([objective_function(np.array([X[i], Y[i]])) for i in range(X.shape[0])])

        ax.plot_surface(X, Y, Z, alpha=.4)

        pos = np.array([p.position for p in particles])
        pos_z = [objective_function(p) for p in pos]

        vel = np.array([p.velocity for p in particles])
        vel_z = [objective_function(p) for p in vel]

        ax.quiver(pos[:, 0], pos[:, 1], pos_z, vel[:, 0],
                  vel[:, 1], vel_z,  alpha=.3, color='black')

        ax.scatter(pos[:, 0], pos[:, 1], pos_z, color='red')

        plt.gca()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])

    anim = matplotlib.animation.FuncAnimation(fig, lambda i: plot_frame(i, hist, ax), frames=range(len(hist)))
    if save_path is not None and filename is not None:
        anim.save(create_path(save_path, filename=filename))

    return anim


def draw_optimization_surface(
        objective_function: ObjectiveFunction,
        bounds: Bound = None,
        x: NDArray[np.float_] = None, y: NDArray[np.float_] = None,
        particles: list[Particle] = [],
        best_position: np.ndarray = None):

    X, Y = _get_surface(bounds, x=x, y=y)

    z = []

    for i in range(X.shape[0]):
        z.append(objective_function(np.array([X[i], Y[i]])))

    Z = np.array(z)

    traces = []

    traces.append(go.Surface(x=X, y=Y, z=Z, colorscale='Blues'))

    if best_position is not None:
        best_z = objective_function(best_position)
        traces.append(go.Scatter3d(x=[best_position[0]], y=[best_position[1]], z=[best_z], mode='markers'))

    if len(particles) > 0:
        pos = np.array([p.position for p in particles])
        pos_z = np.array([objective_function(p) for p in pos])

        vel = np.array([p.velocity for p in particles])
        vel_z = np.array([objective_function(p) for p in vel])

        traces.append(go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos_z, mode='markers'))
        traces.append(go.Cone(
            x=pos[:, 0].tolist(), y=pos[:, 1].tolist(), z=pos_z.tolist(),
            u=vel[:, 0].tolist(), v=vel[:, 1].tolist(), w=vel_z.tolist(),
        ))

    return traces


def plot_surface(
        objective_function: ObjectiveFunction,
        bounds: Bound = None,
        x: NDArray[np.float_] = None, y: NDArray[np.float_] = None,
        particles: list[Particle] = [],
        best_position: np.ndarray = None):

    go_list = draw_optimization_surface(
        objective_function=objective_function,
        bounds=bounds,
        x=x,
        y=y,
        particles=particles,
        best_position=best_position
    )

    fig = go.Figure(data=go_list)
    fig.show()


if __name__ == '__main__':
    x = None
    y = None

    def objective_function(vector: np.ndarray):
        return sum(vector ** 2)

    bounds: Bound = [
        [-100, 100],
        [-100, 100],
    ]

    particles = [Particle(bounds) for _ in range(10)]

    plot_surface(objective_function, bounds, particles=particles)
