from typing import Callable, Union
import numpy as np

from numpy.typing import NDArray


NumericType = Union[int, float]
Bound = list[list[NumericType]]
ObjectiveFunction = Callable[[NDArray[np.float_]], float]
