

import copy

import numpy as np
from numpy.typing import NDArray

from py_pso import Particle
from py_pso.dtypes import Bound, ObjectiveFunction


class PSO:
    bound: Bound
    n_particles: int
    objective_function: ObjectiveFunction

    w: float
    c1: float
    c2: float

    best_score: float
    best_position: NDArray[np.float64]
    particles: list[Particle]

    def __init__(
            self, objective_function: ObjectiveFunction, n_particles: int = 50, bound: Bound = None,
            w: float = .5, c1: float = .8, c2: float = .9) -> None:

        if bound is None:
            bound = [[-1, 1],
                     [-1, 1]]

        self.bound = bound
        self.n_particles = n_particles
        self.objective_function = objective_function

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.best_score = float('inf')
        self.particles = [Particle(bound=bound) for i in range(n_particles)]

    def set_particle_best(self):
        for p in self.particles:
            score = self.objective_function(p.position)
            p.score = score

            if p.score < p.best_score:
                p.best_score = p.score
                p.best_position = p.position

    def set_global_best(self):
        for p in self.particles:
            if p.score < self.best_score:
                self.best_score = p.score
                self.best_position = p.position

    def move_particles(self):
        for p in self.particles:
            inertia = (self.w * p.velocity)
            cognitive_component = self.c1 * np.random.rand() * (p.best_position - p.position)
            social_component = self.c2 * np.random.rand() * (self.best_position - p.position)

            p.velocity = inertia + cognitive_component + social_component
            p.move()

    def iteration(self):
        self.set_particle_best()
        self.set_global_best()
        self.move_particles()

    def optimize(self, iteration: int = 50, seed: int = None, return_hist: bool = False):
        if seed is not None:
            np.random.seed(seed)

        hist = []
        for _ in range(iteration):
            self.iteration()

            if return_hist:
                hist.append(copy.deepcopy(self.particles))

        if return_hist:
            return self.best_position, self.best_score, hist

        return self.best_position, self.best_score
