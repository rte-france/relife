from typing import Literal, Self, overload, Optional

import numpy as np
from numpy.typing import NDArray

from ._plot import PlotKaplanMeier, PlotNelsonAalen, PlotECDF

class ECDF:
    def __init__(self) -> None: ...
    def fit(
        self,
        time: NDArray[np.float64],
    ) -> Self: ...
    @overload
    def sf(
        self, se: Literal[False] = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def sf(
        self, se: Literal[True] = True
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None
    ): ...
    @overload
    def cdf(
        self, se: Literal[False] = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def cdf(
        self, se: Literal[True] = True
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None
    ): ...
    @property
    def plot(self) -> PlotECDF: ...

class KaplanMeier:
    def __init__(self) -> None: ...
    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
    ) -> Self: ...
    @overload
    def sf(
        self, se: Literal[False] = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def sf(
        self, se: Literal[True] = True
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None
    ): ...
    @property
    def plot(self) -> PlotKaplanMeier: ...

class NelsonAalen:
    def __init__(self) -> None: ...
    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.float64]] = None,
        entry: Optional[NDArray[np.float64]] = None,
    ) -> Self: ...
    @overload
    def chf(
        self, se: Literal[False] = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None: ...
    @overload
    def chf(
        self, se: Literal[True] = True
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None
    ): ...
    @property
    def plot(self) -> PlotNelsonAalen: ...
