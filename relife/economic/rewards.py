from functools import partial
from typing import Callable, NewType

import numpy as np
from numpy.typing import NDArray

Rewards = NewType(
    "Rewards",
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
)


def _age_replacement_rewards(
    durations: NDArray[np.float64],
    ar: float | NDArray[np.float64],
    cf: float | NDArray[np.float64],
    cp: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.where(durations < ar, cf, cp)


def age_replacement_rewards(
    ar: float | NDArray[np.float64],
    cf: float | NDArray[np.float64],
    cp: float | NDArray[np.float64],
) -> Rewards:
    return partial(_age_replacement_rewards, ar=ar, cf=cf, cp=cp)


def _run_to_failure_rewards(
    durations: NDArray[np.float64],
    cf: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.ones_like(durations) * cf


def run_to_failure_rewards(cf: float | NDArray[np.float64]) -> Rewards:
    return partial(_run_to_failure_rewards, cf=cf)
