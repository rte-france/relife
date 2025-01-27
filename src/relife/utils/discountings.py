from abc import ABC, abstractmethod
from typing import Generic

import numpy as np
from numpy.typing import NDArray

from relife.utils.types import VariadicArgs


class Discounting(Generic[*VariadicArgs], ABC):
    @abstractmethod
    def factor(
        self, time: NDArray[np.float64], *args: *VariadicArgs
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def rate(
        self, time: NDArray[np.float64], *args: *VariadicArgs
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def annuity_factor(
        self, time: NDArray[np.float64], *args: *VariadicArgs
    ) -> NDArray[np.float64]: ...


class ExponentialDiscounting(Discounting[float]):
    def factor(
        self,
        time: NDArray[np.float64],
        rate: float = 0.0,
    ) -> NDArray[np.float64]:
        return np.exp(-rate * time)

    def rate(
        self,
        time: NDArray[np.float64],
        rate: float = 0.0,
    ) -> NDArray[np.float64]:
        return rate * np.ones_like(time)

    def annuity_factor(
        self,
        time: NDArray[np.float64],
        rate: float = 0.0,
    ) -> NDArray[np.float64]:
        mask = rate == 0
        rate = np.ma.MaskedArray(rate, mask)
        return np.where(mask, time, (1 - np.exp(-rate * time)) / rate)


exponential_discounting = ExponentialDiscounting()
