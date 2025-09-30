from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    Self,
    Union,
    overload,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds

from relife._io_types import (
    _AdditionalIntOrFloatValues,
    _BooleanValues,
    _IntOrFloatValues,
    _NumpyFloatValues,
)
from relife.base import FrozenParametricModel, ParametricModel
from relife.likelihood import FittingResults

from ._plot import PlotParametricLifetimeModel

__all__ = [
    "ParametricLifetimeModel",
    "FittableParametricLifetimeModel",
    "NonParametricLifetimeModel",
    "is_lifetime_model",
]

class ParametricLifetimeModel(ParametricModel, ABC, Generic[*_AdditionalIntOrFloatValues]):
    @abstractmethod
    def sf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    @abstractmethod
    def hf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    @abstractmethod
    def chf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    @abstractmethod
    def pdf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def cdf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def ppf(self, probability: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def median(self, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def isf(self, probability: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def ichf(
        self, cumulative_hazard_rate: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    def moment(self, n: int, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def mean(self, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def var(self, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def mrl(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def ls_integrate(
        self,
        func: Callable[[_IntOrFloatValues], _NumpyFloatValues],
        a: _IntOrFloatValues,
        b: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        deg: int = 10,
    ) -> _NumpyFloatValues: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> _NumpyFloatValues: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _BooleanValues]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _NumpyFloatValues]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _BooleanValues, _NumpyFloatValues]: ...
    def rvs(
        self,
        size: int,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> (
        _NumpyFloatValues
        | tuple[_NumpyFloatValues, _NumpyFloatValues]
        | tuple[_NumpyFloatValues, _BooleanValues]
        | tuple[_NumpyFloatValues, _BooleanValues, _NumpyFloatValues]
    ): ...
    @property
    def plot(self) -> PlotParametricLifetimeModel: ...

class FittableParametricLifetimeModel(ParametricLifetimeModel[*_AdditionalIntOrFloatValues], ABC):
    _fitting_results = Optional[FittingResults]

    def __init__(self, **kwparams: Optional[float]) -> None: ...
    @property
    def fitting_results(self) -> Optional[FittingResults]: ...
    @abstractmethod
    def dhf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    @overload
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[False] = False,
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[True] = True,
    ) -> _NumpyFloatValues: ...
    @abstractmethod
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[False] = False,
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[True] = True,
    ) -> _NumpyFloatValues: ...
    @abstractmethod
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def jac_sf(
        self,
        time: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[False] = False,
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_sf(
        self,
        time: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[True] = True,
    ) -> _NumpyFloatValues: ...
    @abstractmethod
    def jac_sf(
        self,
        time: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def jac_pdf(
        self,
        time: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[False] = False,
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_pdf(
        self,
        time: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[True] = True,
    ) -> _NumpyFloatValues: ...
    @abstractmethod
    def jac_pdf(
        self,
        time: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @abstractmethod
    def _get_params_bounds(self) -> Bounds: ...
    @abstractmethod
    def _get_initial_params(
        self,
        time: NDArray[np.float64],
        *args: *_AdditionalIntOrFloatValues,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]: ...
    def fit(
        self,
        time: NDArray[np.float64],
        *args: *_AdditionalIntOrFloatValues,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        **options: Any,
    ) -> None: ...

class NonParametricLifetimeModel(ABC):
    @abstractmethod
    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> Self: ...

def is_lifetime_model(model: Union[ParametricModel, NonParametricLifetimeModel, FrozenParametricModel]) -> bool: ...
