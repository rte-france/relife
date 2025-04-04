from typing import Any, Callable, Optional, Protocol, Self, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

from .frozen_model import FrozenParametricLifetimeModel

Args = TypeVarTuple("Args")


class FittableParametricLifetimeModel(Protocol[*Args]):
    frozen: bool

    def hf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]: ...

    def chf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]: ...

    def sf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]: ...

    def pdf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]: ...

    def mrl(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]: ...

    def moment(self, n: int, *args: *Args) -> NDArray[np.float64]: ...

    def mean(self, *args: *Args) -> NDArray[np.float64]: ...

    def var(self, *args: *Args) -> NDArray[np.float64]: ...

    def isf(
        self, probability: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]: ...

    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]: ...

    def cdf(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]: ...

    def rvs(
        self, *args: *Args, size: int = 1, seed: Optional[int] = None
    ) -> NDArray[np.float64]: ...

    def ppf(
        self, probability: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]: ...

    def median(self, *args: *Args) -> NDArray[np.float64]: ...

    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        *args: *Args,
        deg: int = 100,
    ) -> NDArray[np.float64]: ...

    def freeze(
        self,
        *args: *Args,
    ) -> FrozenParametricLifetimeModel: ...

    def fit(
        self,
        time: float | NDArray[np.float64],
        /,
        *args: *Args,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> Self: ...
