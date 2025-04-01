from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class Discounting(ABC):
    @abstractmethod
    def factor(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def rate(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def annuity_factor(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]: ...


class ExponentialDiscounting:
    def __init__(self, rate: Optional[float] = None):
        self.rate = rate

    def factor(
        self,
        timeline: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.rate is not None and self.rate != 0.0:
            return np.exp(-self.rate * timeline)
        else:
            return np.ones_like(timeline)

    def annuity_factor(
        self,
        timeline: NDArray[np.float64],
    ) -> NDArray[np.float64]:

        if self.rate is not None and self.rate != 0.0:
            return (1 - np.exp(-self.rate * timeline)) / self.rate
        else:
            return timeline


def exponential_discounting(rate: Optional[float] = None) -> Discounting:
    return ExponentialDiscounting(rate)
