from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from relife.lifetime_model import ParametricLifetimeModel


def cost(
    cf: float | NDArray[np.float64] = 0.,
    cp: float | NDArray[np.float64] = 0.,
    cr: float | NDArray[np.float64] = 0.,
) -> Optional[NDArray[np.float64]]:
    dtype = np.dtype([
        ("cf", np.float64),
        ("cp", np.float64),
        ("cr", np.float64),
    ])
    kwargs = {"cf" : np.asarray(cf), "cp" : np.asarray(cp), "cr" : np.asarray(cr)}
    nb_assets = max(v.size if v.ndim > 0 else 0 for v in kwargs.values())
    shape = (nb_assets, 1) if nb_assets > 0 else ()
    struct_cost = np.zeros(shape, dtype=dtype)
    for k, v in kwargs.items():
        if v.ndim > 0:
            v = v.reshape(-1, 1)
        struct_cost[k] = v
    return struct_cost


class Reward(ABC):
    @abstractmethod
    def conditional_expectation(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        """Conditional expected reward"""
        pass

    @abstractmethod
    def sample(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reward conditional sampling."""


class RunToFailureReward(Reward):
    def __init__(self, cf : float | NDArray[np.float64]):
        self.cost = cost(cf=cf)

    def conditional_expectation(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.ones_like(time) * self.cost["cf"]

    def sample(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.conditional_expectation(time)


class AgeReplacementReward(Reward):
    def __init__(self, cf : float | NDArray[np.float64], cp : float | NDArray[np.float64], ar: float| NDArray[np.float64]):
        self.cost = cost(cf=cf, cp=cp)
        ar = np.asarray(ar, dtype=np.float64)
        shape = () if ar.ndim == 0 else (ar.size, 1)
        self.ar = ar.reshape(shape)

    def conditional_expectation(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(time < self.ar, self.cost["cf"], self.cost["cp"])

    def sample(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.conditional_expectation(time)


def reward_partial_expectation(
    timeline: NDArray[np.float64],
    model: ParametricLifetimeModel[()],
    reward: Reward,
    *,
    discounting: Optional[Discounting] = None,
) -> NDArray[np.float64]:
    def func(x):
        return reward.sample(x) * discounting.factor(x)

    ls = model.ls_integrate(func, np.zeros_like(timeline), timeline)
    # reshape 2d -> final_dim
    ndim = max(map(np.ndim, (timeline, *model.args)), default=0)
    if ndim < 2:
        ls = np.squeeze(ls)
    return ls


class Discounting(Protocol):
    rate: float

    @abstractmethod
    def factor(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def annuity_factor(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]: ...


class ExponentialDiscounting:
    rate: float

    def __init__(self, rate: Optional[float] = None):
        if rate < 0.0:
            raise ValueError
        self.rate = rate if rate is not None else 0.0

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
