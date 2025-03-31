from typing import (
    Protocol,
    Optional,
    Callable,
    TypeVarTuple,
    NewType,
    runtime_checkable,
    Any,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds

from relife.likelihood import LifetimeData
from relife.likelihood.mle import FittingResults
from relife.model.frozen import FrozenLifetimeModel
from relife.nonparametric_model.nonparametrics import Estimates
from relife.plots import PlotSurvivalFunc

Ts = TypeVarTuple("Ts")
T = NewType("T", NDArray[np.floating] | NDArray[np.integer] | float | int)
ModelArgs = NewType(
    "ModelArgs", NDArray[np.floating] | NDArray[np.integer] | float | int
)


class LifetimeModel(Protocol[*Ts]):

    frozen: bool

    def hf(self, time: T, *args: *Ts) -> NDArray[np.float64]: ...

    def chf(self, time: T, *args: *Ts) -> NDArray[np.float64]: ...

    def sf(self, time: T, *args: *Ts) -> NDArray[np.float64]: ...

    def pdf(self, time: T, *args: *Ts) -> NDArray[np.float64]: ...

    def mrl(self, time: T, *args: *Ts) -> NDArray[np.float64]: ...

    def moment(self, n: int, *args: *Ts) -> NDArray[np.float64]: ...

    def mean(self, *args: *Ts) -> NDArray[np.float64]: ...

    def var(self, *args: *Ts) -> NDArray[np.float64]: ...

    def isf(
        self, probability: float | NDArray[np.float64], *args: *Ts
    ) -> NDArray[np.float64]: ...

    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64], *args: *Ts
    ) -> NDArray[np.float64]: ...

    def cdf(self, time: T, *args: *Ts) -> NDArray[np.float64]: ...

    def rvs(
        self, *args: *Ts, size: int = 1, seed: Optional[int] = None
    ) -> NDArray[np.float64]: ...

    def ppf(
        self, probability: float | NDArray[np.float64], *args: *Ts
    ) -> NDArray[np.float64]: ...

    def median(self, *args: *Ts) -> NDArray[np.float64]: ...

    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        *args: *Ts,
        deg: int = 100,
    ) -> NDArray[np.float64]: ...

    def freeze(
        self, **kwargs: ModelArgs
    ) -> FrozenLifetimeModel: ...  # UnivariateLifetimeDistribution


# Only to differentiate those that can be fit from the others
@runtime_checkable
class ParametricLifetimeModel(LifetimeModel[*Ts], Protocol):

    def init_params(self, lifetime_data: LifetimeData, *args: *Ts) -> None: ...

    @property
    def params_bounds(self) -> Bounds: ...

    def fit(
        self,
        time: T,
        /,
        *args: *Ts,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> FittingResults: ...


class NonParametricModel(Protocol):
    """
    Non-parametric_model lifetime estimator.

    Attributes
    ----------
    estimates : Estimations
        The estimations produced when fitting the estimator.
    """

    estimates: dict[str, Optional[Estimates]]

    @property
    def plot(self):
        return PlotSurvivalFunc(self)
