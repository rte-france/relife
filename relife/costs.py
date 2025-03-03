from functools import partial

import numpy as np
from numpy.typing import NDArray


def _age_replacement_cost(
    lifetimes: NDArray[np.float64],
    ar: NDArray[np.float64],
    cf: NDArray[np.float64],
    cp: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.where(lifetimes < ar, cf, cp)


def age_replacement_cost(
    ar: NDArray[np.float64],
    cf: NDArray[np.float64],
    cp: NDArray[np.float64],
):
    return partial(_age_replacement_cost, ar=ar, cf=cf, cp=cp)


def _run_to_failure_cost(
    lifetimes: NDArray[np.float64], cf: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.ones_like(lifetimes) * cf


def run_to_failure_cost(cf: NDArray[np.float64]):
    return partial(_run_to_failure_cost, cf=cf)
