# pyright: basic

from abc import ABC, abstractmethod
from typing import TypeAlias, TypeVarTuple

import numpy as np
from numpy.typing import NDArray
from optype.numpy import Array1D, ArrayND

from relife.utils import to_numpy_float

Ts = TypeVarTuple("Ts")
ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


class Reward(ABC):
    @abstractmethod
    def conditional_expectation(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | NDArray[np.float64]:
        """Conditional expected reward"""
        pass

    @abstractmethod
    def sample(self, time: ST | NumpyST | ArrayND[NumpyST]) -> NDArray[np.float64]:
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

    cf: np.float64 | ArrayND[np.float64]

    def __init__(self, cf: ST | NumpyST | ArrayND[NumpyST]) -> None:
        self.cf = to_numpy_float(cf)

    def conditional_expectation(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | NDArray[np.float64]:
        if isinstance(time, np.ndarray):
            return np.ones_like(time) * self.cf
        return self.cf

    def sample(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | NDArray[np.float64]:
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

    cf: np.float64 | ArrayND[np.float64]
    cp: np.float64 | ArrayND[np.float64]
    ar: np.float64 | ArrayND[np.float64]

    def __init__(
        self,
        cf: int | float | Array1D[np.float64],
        cp: int | float | Array1D[np.float64],
        ar: int | float | Array1D[np.float64],
    ) -> None:
        self.cf = to_numpy_float(cf)
        self.cp = to_numpy_float(cp)
        self.ar = to_numpy_float(ar)

    def conditional_expectation(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | NDArray[np.float64]:
        return np.where(time < self.ar, self.cf, self.cp)

    def sample(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | NDArray[np.float64]:
        return self.conditional_expectation(time)


class Discounting(ABC):
    @abstractmethod
    def factor(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | NDArray[np.float64]: ...

    @abstractmethod
    def annuity_factor(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | NDArray[np.float64]: ...


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

    def factor(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | NDArray[np.float64]:
        if self.rate != 0.0:
            return np.exp(-self.rate * time, dtype=np.float64)
        if isinstance(time, np.ndarray):
            return np.ones_like(time, dtype=np.float64)
        return np.float64(1)

    def annuity_factor(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | NDArray[np.float64]:
        if self.rate != 0.0:
            return (1 - np.exp(-self.rate * time, dtype=np.float64)) / self.rate
        if isinstance(time, np.ndarray):
            return time.astype(np.float64)
        return np.float64(time)
