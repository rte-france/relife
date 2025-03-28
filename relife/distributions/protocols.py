from typing import Protocol, Optional, Callable, TypeVarTuple, NewType

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds

from relife.data import LifetimeData
from relife.nonparametrics import Estimates
from relife.plots import PlotSurvivalFunc

Z = TypeVarTuple("Z")
T = NewType("T", NDArray[np.floating] | NDArray[np.integer] | float | int)


class LifetimeDistribution(Protocol[*Z]):

    def hf(self, time: T, *z: *Z) -> NDArray[np.float64]: ...

    def chf(self, time: T, *z: *Z) -> NDArray[np.float64]: ...

    def sf(self, time: T, *z: *Z) -> NDArray[np.float64]: ...

    def pdf(self, time: T, *z: *Z) -> NDArray[np.float64]: ...

    def mrl(self, time: T, *z: *Z) -> NDArray[np.float64]: ...

    def moment(self, n: int, *z: *Z) -> NDArray[np.float64]: ...

    def mean(self, *z: *Z) -> NDArray[np.float64]: ...

    def var(self, *z: *Z) -> NDArray[np.float64]: ...

    def isf(
        self, probability: float | NDArray[np.float64], *z: *Z
    ) -> NDArray[np.float64]: ...

    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64], *z: *Z
    ) -> NDArray[np.float64]: ...

    def cdf(self, time: T, *z: *Z) -> NDArray[np.float64]: ...

    def rvs(
        self, *z: *Z, size: int = 1, seed: Optional[int] = None
    ) -> NDArray[np.float64]: ...

    def ppf(
        self, probability: float | NDArray[np.float64], *z: *Z
    ) -> NDArray[np.float64]: ...

    def median(self, *z: *Z) -> NDArray[np.float64]: ...

    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        *z: *Z,
        deg: int = 100,
    ) -> NDArray[np.float64]: ...


class ParametricLifetimeDistribution(LifetimeDistribution[*Z], Protocol):
    params: NDArray[np.float64]

    @property
    def nb_params(self) -> int: ...

    @property
    def params_names(self) -> tuple[str, ...]: ...

    def init_params(self, lifetime_data: LifetimeData, *z: *Z) -> None: ...

    @property
    def params_bounds(self) -> Bounds: ...

    def jac_hf(self, time: T, *z: *Z) -> NDArray[np.float64]: ...

    def jac_chf(self, time: T, *z: *Z) -> NDArray[np.float64]: ...

    def dhf(self, time: T, *z: *Z) -> NDArray[np.float64]: ...


class NonParametricModel(Protocol):
    """
    Non-parametric lifetime estimator.

    Attributes
    ----------
    estimates : Estimations
        The estimations produced when fitting the estimator.
    """

    estimates: dict[str, Optional[Estimates]]

    @property
    def plot(self):
        return PlotSurvivalFunc(self)
