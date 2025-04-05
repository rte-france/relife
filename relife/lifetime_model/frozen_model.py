from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, Optional, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

from relife import FrozenParametricModel

if TYPE_CHECKING:
    from relife.lifetime_model._base import ParametricLifetimeModel

Args = TypeVarTuple("Args")


def isbroadcastable(argname: str):
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, x):
            x = np.asarray(x)
            if x.size == 1:
                x = x.item()
            elif x.ndim == 2:
                if x.shape[0] != 1 and x.shape[0] != self.nb_assets:
                    raise ValueError(
                        f"Inconsistent {argname} shape. Got {self.nb_assets} nb of assets but got {x.shape} {argname} shape"
                    )
            return method(self, x)

        return wrapper

    return decorator


class FrozenParametricLifetimeModel(FrozenParametricModel):
    model: ParametricLifetimeModel[*tuple[float | NDArray, ...]]

    @isbroadcastable("time")
    def hf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.hf(time, *self.args)

    @isbroadcastable("time")
    def chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.chf(time, *self.args)

    @isbroadcastable("time")
    def sf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.sf(time, *self.args)

    @isbroadcastable("time")
    def pdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.pdf(time, *self.args)

    @isbroadcastable("time")
    def mrl(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.mrl(time, *self.args)

    def moment(self, n: int) -> NDArray[np.float64]:
        return self.model.moment(n)

    def mean(self) -> NDArray[np.float64]:
        return self.model.moment(1, *self.args)

    def var(self) -> NDArray[np.float64]:
        return self.model.moment(2, *self.args) - self.model.moment(1, *self.args) ** 2

    @isbroadcastable("probability")
    def isf(self, probability: float | NDArray[np.float64]):
        return self.model.isf(probability, *self.args)

    @isbroadcastable("cumulative_hazard_rate")
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]):
        return self.model.ichf(cumulative_hazard_rate, *self.args)

    @isbroadcastable("time")
    def cdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.cdf(time, *self.args)

    def rvs(self, size: int = 1, seed: Optional[int] = None) -> NDArray[np.float64]:
        return self.model.rvs(*self.args, size=size, seed=seed)

    @isbroadcastable("probability")
    def ppf(self, probability: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.ppf(probability, *self.args)

    def median(self) -> NDArray[np.float64]:
        return self.model.median(*self.args)

    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        deg: int = 100,
    ) -> NDArray[np.float64]:

        return self.model.ls_integrate(func, a, b, *self.args, deg=deg)
