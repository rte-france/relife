from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

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



def _check_in_shape(name : str, value : float | NDArray[np.float64], nb_assets : int):
    value = np.asarray(value)
    if value.ndim > 2:
        raise ValueError
    if value.ndim == 2:
        value_nb_assets = value.shape[0]
        if nb_assets != 1:
            if value_nb_assets != 1 and value_nb_assets != nb_assets:
                raise ValueError(f"Incorrect {name} shape. Got {value.shape}, meaning {value_nb_assets} nb_assets but args have {nb_assets} nb assets")
    return value


# using Mixin class allows to preserve same type : FrozenLifetimeDistribtuion := ParametricLifetimeModel[()]
class FrozenParametricLifetimeModel(ParametricLifetimeModel[()], FrozenMixin):

    baseline: ParametricLifetimeModel[*tuple[float | NDArray, ...]]

    def __init__(self, model: ParametricLifetimeModel[*tuple[float | NDArray, ...]]):
        super().__init__()
        self.baseline = model

    @override
    @property
    def args_names(self) -> tuple[()]:
        return ()

    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.hf(time, *self.args)

    def chf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.chf(time, *self.args)

    def sf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.sf(time, *self.args)

    def pdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.pdf(time, *self.args)

    @override
    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        time = _check_in_shape("time", time, self.args_nb_assets)
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
        probability = _check_in_shape("probability", probability, self.args_nb_assets)
        return self.baseline.isf(probability, *self.args)

    @override
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        cumulative_hazard_rate = _check_in_shape("cumulative_hazard_rate", cumulative_hazard_rate, self.args_nb_assets)
        return self.baseline.ichf(cumulative_hazard_rate, *self.args)

    @override
    def cdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.cdf(time, *self.args)

    @override
    def rvs(
        self, size: int | tuple[int, int] = 1, seed: Optional[int] = None
    ) -> np.float64 | NDArray[np.float64]:
        rs = np.random.RandomState(seed=seed)
        match size:
            case int() if size == 1:
                probability = rs.uniform()
            case int() if size > 1:
                probability = rs.uniform(size=(size,))
            case (n,):
                probability = rs.uniform(size=(n,))
            case (m, n):
                if self.args_nb_assets != 1:
                    if m != 1 and m != self.args_nb_assets:
                        raise ValueError(f"Incorrect size. Given args have {self.args_nb_assets} nb assets but size is {size}")
                probability = rs.uniform(size=(m,n))
            case _:
                raise ValueError(f"Incorrect size. Must be int or tuple with no more than 2 elements. Got {size}" )
        return self.isf(probability)

    @override
    def ppf(self, probability: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        probability = _check_in_shape("probability", probability, self.args_nb_assets)
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
        a = _check_in_shape("a", a, self.args_nb_assets)
        b = _check_in_shape("b", b, self.args_nb_assets)
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
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.dhf(time, self.args[0], *self.args[1:])

    def jac_hf(
        self,
        time: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.jac_hf(time, self.args[0], *self.args[1:])

    def jac_chf(
        self,
        time: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.jac_chf(time, self.args[0], *self.args[1:])

    def jac_sf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.jac_sf(time, self.args[0], *self.args[1:])

    def jac_cdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.jac_cdf(time, self.args[0], *self.args[1:])

    def jac_pdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:
        time = _check_in_shape("time", time, self.args_nb_assets)
        return self.baseline.jac_pdf(time, self.args[0], *self.args[1:])
