from typing import Literal, Self, overload

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray

from relife.data import LifetimeData as LifetimeData
from relife.lifetime_model._plot import PlotECDF as PlotECDF
from relife.lifetime_model._plot import PlotKaplanMeier as PlotKaplanMeier
from relife.lifetime_model._plot import PlotNelsonAalen as PlotNelsonAalen
from relife.lifetime_model._plot import PlotTurnbull as PlotTurnbull

from ._base import NonParametricLifetimeModel as NonParametricLifetimeModel

class ECDF(NonParametricLifetimeModel):
    def __init__(self) -> None: ...
    def fit(
        self,
        time: NDArray[np.float64],
        /,
        event: NDArray[np.float64] | None = None,
        entry: NDArray[np.float64] | None = None,
        departure: NDArray[np.float64] | None = None,
    ) -> Self: ...
    @overload
    def sf(self, se: Literal[False] = False) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def sf(
        self, se: Literal[True] = True
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def cdf(self, se: Literal[False] = False) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def cdf(
        self, se: Literal[True] = True
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None: ...
    @property
    def plot(self) -> PlotECDF: ...

class KaplanMeier(NonParametricLifetimeModel):
    def __init__(self) -> None: ...
    def fit(
        self,
        time: NDArray[np.float64],
        event: NDArray[np.float64] | None = None,
        entry: NDArray[np.float64] | None = None,
        departure: NDArray[np.float64] | None = None,
    ) -> Self: ...
    @overload
    def sf(self, se: Literal[False] = False) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def sf(
        self, se: Literal[True] = True
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None: ...
    @property
    def plot(self) -> PlotKaplanMeier: ...

class NelsonAalen(NonParametricLifetimeModel):
    def __init__(self) -> None: ...
    def fit(
        self,
        time: NDArray[np.float64],
        event: NDArray[np.float64] | None = None,
        entry: NDArray[np.float64] | None = None,
        departure: NDArray[np.float64] | None = None,
    ) -> Self: ...
    @overload
    def chf(self, se: Literal[False] = False) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def chf(
        self, se: Literal[True] = True
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None: ...
    @property
    def plot(self) -> PlotNelsonAalen: ...

class Turnbull(NonParametricLifetimeModel):
    tol: Incomplete
    lowmem: Incomplete
    def __init__(self, tol: float | None = 0.0001, lowmem: bool | None = False) -> None: ...
    def fit(
        self,
        time: NDArray[np.float64],
        event: NDArray[np.float64] | None = None,
        entry: NDArray[np.float64] | None = None,
        departure: NDArray[np.float64] | None = None,
        inplace: bool = False,
    ) -> Self: ...
    def sf(self) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @property
    def plot(self) -> PlotTurnbull: ...
