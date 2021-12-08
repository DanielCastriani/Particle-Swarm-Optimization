from typing import Union

import numpy as np
from numpy.typing import NDArray

from py_pso.dtypes import Bound


class Particle:

    position: NDArray[np.float64]
    score: float

    best_position: NDArray[np.float64]
    best_score: float

    velocity: NDArray[np.float64]

    def __init__(self, bound: Bound) -> None:

        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bound])
        self.best_position = self.position
        self.best_score = float('inf')
        self.score = float('inf')
        self.velocity = np.zeros(len(bound))

    def __str__(self) -> str:
        fmt = '{:.2f}'

        best_pos = f"({', '.join([fmt.format(p) for p in self.best_position])})"
        pos = f"({', '.join([fmt.format(p) for p in self.position])})"
        velocity = f"({', '.join([fmt.format(p) for p in self.velocity])})"

        return f'Current Position: {pos:>30} | Best Position {best_pos:>30} | Velocity: {velocity:>30} | Best Score: {self.best_score:>10}'

    def __repr__(self) -> str:
        return f'<Particle {self.__str__()}>'

    def move(self):
        self.position = self.position + self.velocity
