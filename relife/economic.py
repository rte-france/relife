from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Callable, NewType, Optional, Protocol

import numpy as np
from numpy._typing import NDArray
from numpy.typing import NDArray

if TYPE_CHECKING:
    from relife.lifetime_model import ParametricLifetimeModel


Reward = NewType(
    "Reward",
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
)


class Cost(dict):
    _allowed_keys = ("cp", "cf", "cr")

    def __init__(
        self,
        mapping: Optional[dict[str, float | NDArray[np.float64]]] = None,
        /,
        **kwargs: float | NDArray[np.float64],
    ):
        if mapping is None:
            mapping = {}
        mapping.update(kwargs)
        if not set(mapping.keys()).issubset(self._allowed_keys):
            raise ValueError(f"Only {self._allowed_keys} parameters are allowed")

        nb_assets = 1
        for name, value in mapping.items():
            value = np.asarray(value)
            ndim = value.ndim
            if ndim > 2:
                raise ValueError(
                    f"Number of dimension can't be higher than 2. Got {ndim} for {name}"
                )
            if value.ndim == 2 and value.shape[-1] != 1:
                raise ValueError(
                    f"Incorrect {name} shape. If ar has 2 dim, the shape must be (m, 1) only. Got {value.shape}")
            if value.ndim <= 1:
                value = value.reshape(-1, 1)
            # check if nb_assets is no more compatible
            if nb_assets != 1 and value.shape[0] not in (1, nb_assets):
                raise ValueError(f"Incompatible nb_assets. Got {nb_assets} and {value.shape[0]}")
            # update nb_assets
            if value.shape[0] != 1:
                nb_assets = value.shape[0]
            mapping[name] = value
        self.nb_assets = nb_assets
        super().__init__(mapping)

    def __setitem__(self, key, val):
        raise AttributeError("Can't set item")

    def update(self, *args, **kwargs):
        raise AttributeError("Can't update items")


def _age_replacement_rewards(
    time: NDArray[np.float64], # duration
    ar: float | NDArray[np.float64],
    cf: float | NDArray[np.float64],
    cp: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.where(time < ar, cf, cp)


def age_replacement_rewards(
    ar: float | NDArray[np.float64],
    cf: float | NDArray[np.float64],
    cp: float | NDArray[np.float64],
) -> Reward:
    return partial(_age_replacement_rewards, ar=ar, cf=cf, cp=cp)


def _run_to_failure_rewards(
    time: NDArray[np.float64], # duration
    cf: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.ones_like(time) * cf


def run_to_failure_rewards(cf: float | NDArray[np.float64]) -> Reward:
    return partial(_run_to_failure_rewards, cf=cf)


def reward(ar : Optional[float | NDArray[np.float64]] = None, /, **kwargs : float | NDArray[np.float64]) -> Reward:
    cost = Cost(**kwargs)
    if ar is None:
        if sorted(cost.keys()) != ["cf"]:
            raise ValueError
        return run_to_failure_rewards(cost["cf"])
    else:
        if sorted(cost.keys()) != ["cf", "cp"]:
            raise ValueError
        ar = np.squeeze(np.asarray(ar))
        if ar .ndim > 2:
            raise ValueError(f"Incorrect ar dim. Can't be more than 2. Got {ar.ndim}")
        if ar.ndim == 2 and ar.shape[-1] != 1:
            raise ValueError(
                f"Incorrect ar shape. If ar has 2 dim, the shape must be (m, 1) only. Got {ar.shape}")
        if ar.ndim == 1:
            ar = ar.reshape(-1, 1)
        if ar.shape[0] != 1 and ar.shape[0] != cost.nb_assets:
            raise ValueError
        return age_replacement_rewards(ar, cost["cf"], cost["cp"])


def reward_partial_expectation(
    timeline: NDArray[np.float64],
    model: ParametricLifetimeModel[()],
    rewards_func: Reward,
    *,
    discounting: Optional[Discounting] = None,
) -> NDArray[np.float64]:
    def func(x):
        return rewards_func(x) * discounting.factor(x)

    ls = model.ls_integrate(func, np.zeros_like(timeline), timeline)
    # reshape 2d -> final_dim
    ndim = max(map(np.ndim, (timeline, *model.args)), default=0)
    if ndim < 2:
        ls = np.squeeze(ls)
    return ls


class Discounting(Protocol):
    rate : float

    @abstractmethod
    def factor(self, timeline : NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def annuity_factor(self, timeline : NDArray[np.float64]) -> NDArray[np.float64]: ...


class ExponentialDiscounting:
    rate : float

    def __init__(self, rate: Optional[float] = None):
        if rate < 0.:
            raise ValueError
        self.rate = rate if rate is not None else 0.

    def factor(
        self,
        timeline : NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.rate != 0.0:
            return np.exp(-self.rate * timeline)
        else:
            return np.ones_like(timeline)

    def annuity_factor(
        self,
        timeline : NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.rate != 0.0:
            return (1 - np.exp(-self.rate * timeline)) / self.rate
        else:
            return timeline
