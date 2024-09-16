from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVarTuple

import numpy as np
from numpy.typing import NDArray

Ts = TypeVarTuple("Ts")


class Reward(Callable[*Ts], ABC):

    @abstractmethod
    def __call__(
        self, duration: NDArray[np.float64], *args: *Ts
    ) -> NDArray[np.float64]: ...


class RuntoFailureCost(Reward[NDArray[np.float64]]):
    def __call__(
        self, duration: NDArray[np.float64], cf: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return cf


class AgeReplacementCost(
    Reward[*tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]
):
    def __call__(
        self,
        duration: NDArray[np.float64],
        ar: NDArray[np.float64],
        cf: NDArray[np.float64],
        cp: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.where(duration < ar, cf, cp)
