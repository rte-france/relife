from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    Self,
    TypeVarTuple,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds

if TYPE_CHECKING:
    from relife._plots import PlotConstructor
    from relife.data import LifetimeData

    from ._base import NonParametricEstimation
    from ._frozen import FrozenLifetimeModel


Args = TypeVarTuple("Args")


class LifetimeModel(Protocol[*Args]):

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
    ) -> FrozenLifetimeModel: ...


# Only to differentiate those that can be fit from the others
@runtime_checkable
class ParametricLifetimeModel(LifetimeModel[*Args], Protocol):

    params: NDArray[np.float64]
    params_names: tuple[str, ...]
    components: tuple[Self, ...]

    @property
    def params_names(self) -> tuple[str, ...]: ...

    def init_params(self, lifetime_data: LifetimeData, *args: *Args) -> None: ...

    @property
    def params_bounds(self) -> Bounds: ...

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


@runtime_checkable
class NonParametricLifetimeModel(Protocol):
    estimations: Optional[NonParametricEstimation]

    def __init__(self):
        self.estimations = None

    def fit(
        self,
        time: float | NDArray[np.float64],
        /,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> Self: ...

    def plot(self) -> PlotConstructor: ...
