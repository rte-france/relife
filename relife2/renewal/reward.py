from typing import Protocol, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

Ts = TypeVarTuple("Ts")


class Reward(Protocol[*Ts]):
    def __call__(self, lifetimes: NDArray[np.float64], *args: *Ts): ...


class RunToFailureCost(Reward[NDArray[np.float64]]):
    def __call__(
        self, lifetimes: NDArray[np.float64], cf: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.ones_like(lifetimes) * cf


class AgeReplacementCost(
    Reward[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
):
    def __call__(
        self,
        lifetimes: NDArray[np.float64],
        ar: NDArray[np.float64],
        cf: NDArray[np.float64],
        cp: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.where(lifetimes < ar, cf, cp)


run_to_failure_cost = RunToFailureCost()
age_replacement_cost = AgeReplacementCost()
