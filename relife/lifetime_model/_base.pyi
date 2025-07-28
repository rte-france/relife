import abc
import numpy as np
from scipy.optimize import Bounds

from ._plot import PlotParametricLifetimeModel as PlotParametricLifetimeModel
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from relife import FrozenParametricModel as FrozenParametricModel, ParametricModel as ParametricModel
from relife.likelihood import FittingResults as FittingResults
from typing import Callable, Generic, Literal, Self, TypeVarTuple, overload, Optional, Any

from ..data import LifetimeData

Args = TypeVarTuple("Args")

class ParametricLifetimeModel(ParametricModel, ABC, Generic[*Args]):
    @abstractmethod
    def sf(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]: ...
    @abstractmethod
    def hf(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]: ...
    @abstractmethod
    def chf(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]: ...
    @abstractmethod

    def pdf(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]: ...
    def cdf(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]: ...
    def ppf(self, probability: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]: ...
    def median(self, *args: *Args) -> np.float64 | NDArray[np.float64]: ...
    def isf(self, probability: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]: ...
    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64], *args: *Args
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *Args,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: int | None = None,
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *Args,
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[False],
        seed: int | None = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_]]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *Args,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[True],
        seed: int | None = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *Args,
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[True],
        seed: int | None = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_], np.float64 | NDArray[np.float64]]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *Args,
        nb_assets: int | None = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: int | None = None,
    ) -> (
        np.float64
        | NDArray[np.float64]
        | tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]
        | tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_]]
        | tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_], np.float64 | NDArray[np.float64]]
    ): ...
    @property
    def plot(self) -> PlotParametricLifetimeModel: ...
    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], np.float64 | NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        *args: *Args,
        deg: int = 10,
    ) -> np.float64 | NDArray[np.float64]: ...
    def moment(self, n: int, *args: *Args) -> np.float64 | NDArray[np.float64]: ...
    def mean(self, *args: *Args) -> np.float64 | NDArray[np.float64]: ...
    def var(self, *args: *Args) -> np.float64 | NDArray[np.float64]: ...
    def mrl(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]: ...
    def freeze(self, *args: *Args): ...

class FittableParametricLifetimeModel(ParametricLifetimeModel[*Args], ABC):
    fitting_results = Optional[FittingResults]

    def __init__(self, **kwparams: float | None) -> None: ...

    @abstractmethod
    def _init_params(self, lifetime_data : LifetimeData) -> None: ...

    @abstractmethod
    def _params_bounds(self) -> Bounds: ...

    def fit(
        self,
        time: NDArray[np.float64],
        *args: *Args,
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
        departure: NDArray[np.float64] | None = None,
        **options : Any,
    ) -> Self: ...

class NonParametricLifetimeModel(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def fit(
        self,
        time: NDArray[np.float64],
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
        departure: NDArray[np.float64] | None = None,
    ) -> Self: ...

class FrozenParametricLifetimeModel(FrozenParametricModel[*Args]):

    unfrozen_model : ParametricLifetimeModel[*Args]

    def __init__(
        self, model: ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]], *args: *Args
    ) -> None: ...
    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def chf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def sf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def pdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def moment(self, n: int) -> np.float64 | NDArray[np.float64]: ...
    def mean(self) -> np.float64 | NDArray[np.float64]: ...
    def var(self) -> np.float64 | NDArray[np.float64]: ...
    def isf(self, probability: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def cdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: int | None = None,
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        return_event: Literal[True],
        return_entry: Literal[False],
        seed: int | None = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_]]: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        return_event: Literal[False],
        return_entry: Literal[True],
        seed: int | None = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        return_event: Literal[True],
        return_entry: Literal[True],
        seed: int | None = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_], np.float64 | NDArray[np.float64]]: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int] | tuple[int, int],
        return_event: bool = False,
        return_entry: bool = False,
        seed: int | None = None,
    ) -> (
        np.float64
        | NDArray[np.float64]
        | tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_]]
        | tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]
        | tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_], np.float64 | NDArray[np.float64]]
    ): ...
    def ppf(self, probability: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def median(self) -> np.float64 | NDArray[np.float64]: ...
    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], np.float64 | NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        deg: int = 10,
    ) -> np.float64 | NDArray[np.float64]: ...
