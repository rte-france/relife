from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, NewType, Optional, ParamSpec

import numpy as np
from numpy.typing import NDArray


class Discounting(ABC):
    @abstractmethod
    def factor(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def rate(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def annuity_factor(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]: ...


class ExpDiscounting:
    def __init__(self, rate: Optional[float] = None):
        self.rate = rate

    def factor(
        self,
        timeline: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.rate is not None:
            return 1.0 / np.exp(self.rate * timeline)
        else:
            return np.ones_like(timeline)

    def annuity_factor(
        self,
        timeline: NDArray[np.float64],
    ) -> NDArray[np.float64]:

        if self.rate is not None:
            return (1 - np.exp(self.rate * timeline)) / self.rate
        else:
            return np.ones_like(timeline)


def exp_discounting(rate: Optional[float] = None) -> Discounting:
    return ExpDiscounting(rate)


RewardsFunc = NewType(
    "RewardFunc",
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
)


def _age_replacement_rewards(
    durations: NDArray[np.float64],
    ar: NDArray[np.float64],
    cf: NDArray[np.float64],
    cp: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.where(durations < ar, cf, cp)


def age_replacement_rewards(
    ar: NDArray[np.float64],
    cf: NDArray[np.float64],
    cp: NDArray[np.float64],
) -> RewardsFunc:
    return partial(_age_replacement_rewards, ar=ar, cf=cf, cp=cp)


def _run_to_failure_rewards(
    durations: NDArray[np.float64],
    cf: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.ones_like(durations) * cf


def run_to_failure_rewards(
    cf: NDArray[np.float64], discounting_rate: float = 0.0
) -> RewardsFunc:
    return partial(_run_to_failure_rewards, cf=cf)
