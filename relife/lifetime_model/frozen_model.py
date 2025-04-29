from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, ParamSpec, TypeVarTuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife._base import FrozenMixin

from ._base import ParametricLifetimeModel

if TYPE_CHECKING:
    from relife.lifetime_model import (
        LifetimeDistribution,
        LifetimeRegression,
        ParametricLifetimeModel,
    )

Args = TypeVarTuple("Args")
P = ParamSpec("P")


# using Mixin class allows to preserve same type : FrozenLifetimeDistribtuion := ParametricLifetimeModel[()]
class FrozenParametricLifetimeModel(ParametricLifetimeModel[()], FrozenMixin):

    baseline: ParametricLifetimeModel[*tuple[float | NDArray, ...]]

    def __init__(self, model: ParametricLifetimeModel[*tuple[float | NDArray, ...]]):
        super().__init__()
        self.compose_with(baseline=model)

    @override
    @property
    def args_names(self) -> tuple[()]:
        return ()

    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.baseline.hf(time, *self.args)

    def chf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.baseline.chf(time, *self.args)

    def sf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.baseline.sf(time, *self.args)

    def pdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.baseline.pdf(time, *self.args)

    @override
    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.baseline.mrl(time, *self.args)

    @override
    def moment(self, n: int) -> np.float64 | NDArray[np.float64]:
        return self.baseline.moment(n, *self.args)

    @override
    def mean(self) -> np.float64 | NDArray[np.float64]:
        return self.baseline.moment(1, *self.args)

    @override
    def var(self) -> np.float64 | NDArray[np.float64]:
        return (
            self.baseline.moment(2, *self.args)
            - self.baseline.moment(1, *self.args) ** 2
        )

    @override
    def isf(self, probability: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.baseline.isf(probability, *self.args)

    @override
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.baseline.ichf(cumulative_hazard_rate, *self.args)

    @override
    def cdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.baseline.cdf(time, *self.args)

    @override
    def rvs(
        self, shape: int | tuple[int, int], seed: Optional[int] = None
    ) -> np.float64 | NDArray[np.float64]:
        return self.baseline.rvs(shape, *self.args, seed=seed)

    @override
    def ppf(self, probability: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.baseline.ppf(probability, *self.args)

    @override
    def median(self) -> np.float64 | NDArray[np.float64]:
        return self.baseline.median(*self.args)

    @override
    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        deg: int = 10,
    ) -> np.float64 | NDArray[np.float64]:

        return self.baseline.ls_integrate(func, a, b, *self.args, deg=deg)


class FrozenLifetimeDistribution(FrozenParametricLifetimeModel):
    baseline: LifetimeDistribution

    def dhf(
        self,
        time: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        return self.baseline.dhf(time)


    def jac_hf(
        self,
        time: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.baseline.jac_hf(time)

    def jac_chf(
        self,
        time: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.baseline.jac_chf(time)

    def jac_sf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.baseline.jac_sf(time)

    def jac_cdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.baseline.jac_cdf(time)

    def jac_pdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.baseline.jac_pdf(time)


class FrozenLifetimeRegression(FrozenParametricLifetimeModel):
    baseline: LifetimeRegression

    def dhf(
        self,
        time: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        return self.baseline.dhf(time, self.args[0], *self.args[1:])

    def jac_hf(
        self,
        time: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.baseline.jac_hf(time, self.args[0], *self.args[1:])

    def jac_chf(
        self,
        time: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.baseline.jac_chf(time, self.args[0], *self.args[1:])

    def jac_sf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.baseline.jac_sf(time, self.args[0], *self.args[1:])

    def jac_cdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.baseline.jac_cdf(time, self.args[0], *self.args[1:])

    def jac_pdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        return self.baseline.jac_pdf(time, self.args[0], *self.args[1:])
