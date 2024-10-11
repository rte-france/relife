from typing import Callable, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

RArgs = TypeVarTuple("RArgs")

Reward = Callable[[NDArray[np.float64], *RArgs], NDArray[np.float64]]


def run_to_failure_cost(
    lifetimes: NDArray[np.float64], cf: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.ones_like(lifetimes) * cf


def age_replacement_cost(
    lifetimes: NDArray[np.float64],
    ar: NDArray[np.float64],
    cf: NDArray[np.float64],
    cp: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.where(lifetimes < ar, cf, cp)
