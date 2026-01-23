# pyright: basic

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from relife.typing import AnyFloat
from relife.utils import reshape_1d_arg


class Reward(ABC):
    @abstractmethod
    def conditional_expectation(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        """Conditional expected reward"""
        pass

    @abstractmethod
    def sample(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reward conditional sampling.

        Parameters
        ----------
        time : ndarray
            Time duration values.


        Returns
        -------
        ndarray
            Random drawing of a reward with respect to time.
        """


class RunToFailureReward(Reward):
    r"""Run-to-failure reward.

    Parameters
    ----------
    cf : float or 1darray
        The cost of failure.

    Attributes
    ----------
    cf
    """

    cf: np.float64 | NDArray[np.float64]

    def __init__(self, cf: AnyFloat) -> None:
        self.cf = reshape_1d_arg(cf)

    def conditional_expectation(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.ones_like(time) * self.cf

    def sample(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.conditional_expectation(time)


class AgeReplacementReward(Reward):
    r"""Age replacement reward.

    Parameters
    ----------
    cf : float or 1darray
        The cost of failure.
    cp : float or 1darray
        The cost of preventive replacement.

    Attributes
    ----------
    cf
    cp
    ar
    """

    cf: np.float64 | NDArray[np.float64]
    cp: np.float64 | NDArray[np.float64]
    ar: np.float64 | NDArray[np.float64]

    def __init__(self, cf: AnyFloat, cp: AnyFloat, ar: AnyFloat) -> None:
        self.cf = reshape_1d_arg(cf)
        self.cp = reshape_1d_arg(cp)
        self.ar = reshape_1d_arg(ar)

    def conditional_expectation(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(time < self.ar, self.cf, self.cp)

    def sample(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.conditional_expectation(time)


class Discounting(ABC):
    @abstractmethod
    def factor(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def annuity_factor(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]: ...


class ExponentialDiscounting(Discounting):
    """
    Exponential discounting.

    Parameters
    ----------
    rate : float
        The discounting rate
    """

    rate: float

    def __init__(self, rate: float = 0.0) -> None:
        self.rate = rate

    def factor(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.rate != 0.0:
            return np.exp(-self.rate * timeline)
        else:
            return np.ones_like(timeline)

    def annuity_factor(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.rate != 0.0:
            return (1 - np.exp(-self.rate * timeline)) / self.rate
        else:
            return timeline
