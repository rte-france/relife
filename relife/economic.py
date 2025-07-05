from abc import ABC, abstractmethod
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray


def cost(
    cf: float | NDArray[np.float64] = 0.0,
    cp: float | NDArray[np.float64] = 0.0,
    cr: float | NDArray[np.float64] = 0.0,
) -> NDArray[np.void]:
    struct_dtype = np.dtype(
        [
            ("cf", np.float64),
            ("cp", np.float64),
            ("cr", np.float64),
        ]
    )
    kwargs = {"cf": np.asarray(cf), "cp": np.asarray(cp), "cr": np.asarray(cr)}
    nb_assets = max(v.size if v.ndim > 0 else 0 for v in kwargs.values())
    shape: tuple[int, Literal[1]] | tuple[()] = (nb_assets, 1) if nb_assets > 0 else ()
    struct_cost = np.zeros(shape, dtype=struct_dtype)
    for k, v in kwargs.items():
        if v.ndim > 0:
            v = v.reshape(-1, 1)
        struct_cost[k] = v
    return struct_cost


class Reward(ABC):
    _cost_array: NDArray[np.void]

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

    @property
    def nb_assets(self) -> int:
        """
        Number of assets.
        """
        if self._cost_array.shape == ():
            return 1
        return self._cost_array.shape[0]

    @property
    def ndim(self) -> int:
        """
        Cost number of dimension (either 0 or 1)
        """
        return self._cost_array.ndim

    @property
    def size(self) -> int:
        """
        Cost size
        """
        return self._cost_array.size


class RunToFailureReward(Reward):
    # noinspection PyUnresolvedReferences
    r"""Run-to-failure reward.

    Parameters
    ----------
    cf : float or 1darray
        The cost(s) of failure

    Attributes
    ----------
    cf
    """

    def __init__(self, cf: float | NDArray[np.float64]):
        self._cost_array = cost(cf=cf)

    @property
    def cf(self) -> NDArray[np.float64]:
        """
        Cost of failures

        Returns
        -------
        ndarray
        """
        return self._cost_array["cf"]

    @cf.setter
    def cf(self, value: float | NDArray[np.float64]) -> None:
        self._cost_array["cf"] = value

    def conditional_expectation(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.ones_like(time) * self.cf

    def sample(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.conditional_expectation(time)


class AgeReplacementReward(Reward):
    # noinspection PyUnresolvedReferences
    r"""Age replacement reward.

    Parameters
    ----------
    cf : float or 1darray
        The cost(s) of failure
    cp : float or 1darray
        The cost(s) of preventive replacement

    Attributes
    ----------
    cf
    cp
    ar

    """

    def __init__(
        self, cf: float | NDArray[np.float64], cp: float | NDArray[np.float64], ar: float | NDArray[np.float64]
    ):
        self._cost_array = cost(cf=cf, cp=cp)
        ar = np.asarray(ar, dtype=np.float64)
        shape = () if ar.ndim == 0 else (ar.size, 1)
        self._ar = ar.reshape(shape)

    @property
    def cf(self) -> NDArray[np.float64]:
        """
        Cost of failures

        Returns
        -------
        ndarray
        """
        return self._cost_array["cf"]

    @property
    def cp(self) -> NDArray[np.float64]:
        """
        Cost of preventive replacement

        Returns
        -------
        ndarray
        """
        return self._cost_array["cp"]

    @cf.setter
    def cf(self, value: float | NDArray[np.float64]) -> None:
        self._cost_array["cf"] = value

    @cp.setter
    def cp(self, value: float | NDArray[np.float64]) -> None:
        self._cost_array["cp"] = value

    @property
    def ar(self) -> float | NDArray[np.float64]:
        """
        Age of replacement

        Returns
        -------
        ndarray
        """
        return self._ar

    @ar.setter
    def ar(self, value: float | NDArray[np.float64]) -> None:
        shape = () if value.ndim == 0 else (value.size, 1)
        self._ar = value.reshape(shape)

    def conditional_expectation(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(time < self.ar, self.cf, self.cp)

    def sample(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.conditional_expectation(time)


class Discounting(Protocol):
    rate: float

    @abstractmethod
    def factor(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def annuity_factor(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]: ...


class ExponentialDiscounting:
    rate: float

    def __init__(self, rate: float = 0.0):
        if rate < 0.0:
            raise ValueError(f"Invalid rate value. It must be positive. Got {rate}")
        self.rate = rate

    def factor(
        self,
        timeline: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.rate != 0.0:
            return np.exp(-self.rate * timeline)
        else:
            return np.ones_like(timeline)

    def annuity_factor(
        self,
        timeline: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.rate != 0.0:
            return (1 - np.exp(-self.rate * timeline)) / self.rate
        else:
            return timeline
