from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, ParamSpec, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

from relife import FrozenParametricModel

if TYPE_CHECKING:
    from relife.lifetime_model._base import ParametricLifetimeModel

Args = TypeVarTuple("Args")
P = ParamSpec("P")


# def _isbroadcastable(
#     arg_name: str,
# ) -> Callable[[Callable[[P], NDArray[np.float64]]], Callable[[P], NDArray[np.float64]]]:
#     def decorator(
#         method: Callable[P, NDArray[np.float64]],
#     ) -> Callable[[P], NDArray[np.float64]]:
#         @functools.wraps(method)
#         def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> NDArray[np.float64]:
#             x: NDArray[np.float64] = np.asarray(args[0])
#             if x.size == 1:
#                 x = x.item()
#             elif x.ndim == 2:
#                 if self.nb_assets != 1:
#                     if x.shape[0] != 1 and x.shape[0] != self.nb_assets:
#                         raise ValueError(
#                             f"Inconsistent {arg_name} shape. Got {self.nb_assets} nb of assets but got {x.shape} {arg_name} shape"
#                         )
#             return method(self, x, **kwargs)
#
#         return wrapper
#
#     return decorator


class FrozenParametricLifetimeModel(FrozenParametricModel):
    baseline: ParametricLifetimeModel[*tuple[float | NDArray, ...]]

    # @_isbroadcastable("time")
    def hf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.hf(time, *self.args)

    # @_isbroadcastable("time")
    def chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.chf(time, *self.args)

    # @_isbroadcastable("time")
    def sf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.sf(time, *self.args)

    # @_isbroadcastable("time")
    def pdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.pdf(time, *self.args)

    # @_isbroadcastable("time")
    def mrl(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.mrl(time, *self.args)

    def moment(self, n: int) -> NDArray[np.float64]:
        return self.baseline.moment(n)

    def mean(self) -> NDArray[np.float64]:
        return self.baseline.moment(1, *self.args)

    def var(self) -> NDArray[np.float64]:
        return self.baseline.moment(2, *self.args) - self.baseline.moment(1, *self.args) ** 2

    # @_isbroadcastable("probability")
    def isf(self, probability: float | NDArray[np.float64]):
        return self.baseline.isf(probability, *self.args)

    # @_isbroadcastable("cumulative_hazard_rate")
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]):
        return self.baseline.ichf(cumulative_hazard_rate, *self.args)

    # @_isbroadcastable("time")
    def cdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.cdf(time, *self.args)

    def rvs(self, size: int = 1, seed: Optional[int] = None) -> NDArray[np.float64]:
        return self.baseline.rvs(*self.args, size=size, seed=seed)

    # @_isbroadcastable("probability")
    def ppf(self, probability: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.ppf(probability, *self.args)

    def median(self) -> NDArray[np.float64]:
        return self.baseline.median(*self.args)

    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        deg: int = 100,
    ) -> NDArray[np.float64]:

        return self.baseline.ls_integrate(self, func, a, b, deg=deg)
