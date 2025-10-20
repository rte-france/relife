from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Literal, Optional, Union, overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds

from relife._typing import (
    _Any_Number,
    _Any_Number_Ts,
    _Any_Numpy_Bool,
    _Any_Numpy_Number,
    _NumpyArray_OfBool,
    _NumpyArray_OfNumber,
)
from relife.base import ParametricModel
from relife.likelihood import FittingResults

from ._plot import PlotParametricLifetimeModel

__all__ = [
    "ParametricLifetimeModel",
    "FittableParametricLifetimeModel",
]

class ParametricLifetimeModel(ParametricModel, ABC, Generic[*_Any_Number_Ts]):
    @abstractmethod
    def sf(self, time: _Any_Number, *args: *_Any_Number_Ts) -> _Any_Numpy_Number: ...
    @abstractmethod
    def hf(self, time: _Any_Number, *args: *_Any_Number_Ts) -> _Any_Numpy_Number: ...
    @abstractmethod
    def chf(self, time: _Any_Number, *args: *_Any_Number_Ts) -> _Any_Numpy_Number: ...
    @abstractmethod
    def pdf(self, time: _Any_Number, *args: *_Any_Number_Ts) -> _Any_Numpy_Number: ...
    def cdf(self, time: _Any_Number, *args: *_Any_Number_Ts) -> _Any_Numpy_Number: ...
    def ppf(
        self, probability: _Any_Number, *args: *_Any_Number_Ts
    ) -> _Any_Numpy_Number: ...
    def median(self, *args: *_Any_Number_Ts) -> _Any_Numpy_Number: ...
    def isf(
        self, probability: _Any_Number, *args: *_Any_Number_Ts
    ) -> _Any_Numpy_Number: ...
    def ichf(
        self, cumulative_hazard_rate: _Any_Number, *args: *_Any_Number_Ts
    ) -> _Any_Numpy_Number: ...
    def moment(self, n: int, *args: *_Any_Number_Ts) -> _Any_Numpy_Number: ...
    def mean(self, *args: *_Any_Number_Ts) -> _Any_Numpy_Number: ...
    def var(self, *args: *_Any_Number_Ts) -> _Any_Numpy_Number: ...
    def mrl(self, time: _Any_Number, *args: *_Any_Number_Ts) -> _Any_Numpy_Number: ...
    def ls_integrate(
        self,
        func: Callable[[_Any_Number], _Any_Numpy_Number],
        a: _Any_Number,
        b: _Any_Number,
        *args: *_Any_Number_Ts,
        deg: int = 10,
    ) -> _Any_Numpy_Number: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_Any_Number_Ts,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> _Any_Numpy_Number: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_Any_Number_Ts,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> tuple[_Any_Numpy_Number, _Any_Numpy_Bool]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_Any_Number_Ts,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> tuple[_Any_Numpy_Number, _Any_Numpy_Number]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_Any_Number_Ts,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> tuple[_Any_Numpy_Number, _Any_Numpy_Bool, _Any_Numpy_Number]: ...
    def rvs(
        self,
        size: int,
        *args: *_Any_Number_Ts,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> (
        _Any_Numpy_Number
        | tuple[_Any_Numpy_Number, _Any_Numpy_Number]
        | tuple[_Any_Numpy_Number, _Any_Numpy_Bool]
        | tuple[_Any_Numpy_Number, _Any_Numpy_Bool, _Any_Numpy_Number]
    ): ...
    @property
    def plot(self) -> PlotParametricLifetimeModel: ...

class FittableParametricLifetimeModel(ParametricLifetimeModel[*_Any_Number_Ts], ABC):
    fitting_results = Optional[FittingResults]

    def __init__(self, **kwparams: Optional[float]) -> None: ...
    @abstractmethod
    def dhf(self, time: _Any_Number, *args: *_Any_Number_Ts) -> _Any_Numpy_Number: ...
    @overload
    def jac_hf(
        self,
        time: _Any_Number,
        *args: *_Any_Number_Ts,
        asarray: Literal[False] = False,
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _Any_Number,
        *args: *_Any_Number_Ts,
        asarray: Literal[True] = True,
    ) -> _Any_Numpy_Number: ...
    @abstractmethod
    def jac_hf(
        self,
        time: _Any_Number,
        *args: *_Any_Number_Ts,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def jac_chf(
        self,
        time: _Any_Number,
        *args: *_Any_Number_Ts,
        asarray: Literal[False] = False,
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _Any_Number,
        *args: *_Any_Number_Ts,
        asarray: Literal[True] = True,
    ) -> _Any_Numpy_Number: ...
    @abstractmethod
    def jac_chf(
        self,
        time: _Any_Number,
        *args: *_Any_Number_Ts,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def jac_sf(
        self,
        time: _Any_Number,
        *args: *_Any_Number_Ts,
        asarray: Literal[False] = False,
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_sf(
        self,
        time: _Any_Number,
        *args: *_Any_Number_Ts,
        asarray: Literal[True] = True,
    ) -> _Any_Numpy_Number: ...
    @abstractmethod
    def jac_sf(
        self,
        time: _Any_Number,
        *args: *_Any_Number_Ts,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def jac_pdf(
        self,
        time: _Any_Number,
        *args: *_Any_Number_Ts,
        asarray: Literal[False] = False,
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_pdf(
        self,
        time: _Any_Number,
        *args: *_Any_Number_Ts,
        asarray: Literal[True] = True,
    ) -> _Any_Numpy_Number: ...
    @abstractmethod
    def jac_pdf(
        self,
        time: _Any_Number,
        *args: *_Any_Number_Ts,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @abstractmethod
    def _get_params_bounds(self) -> Bounds: ...
    @abstractmethod
    def _get_initial_params(
        self,
        time: _NumpyArray_OfNumber,
        *args: *_Any_Number_Ts,
        event: Optional[_NumpyArray_OfBool] = None,
        entry: Optional[_NumpyArray_OfNumber] = None,
    ) -> NDArray[np.float64]: ...
    def fit(
        self,
        time: _NumpyArray_OfNumber,
        *args: *_Any_Number_Ts,
        event: Optional[_NumpyArray_OfBool] = None,
        entry: Optional[_NumpyArray_OfNumber] = None,
        optimizer_options: Optional[dict[str, Any]] = None,
    ) -> None: ...
    def fit_from_interval_censored_lifetimes(
        self,
        time_inf: _NumpyArray_OfNumber,
        time_sup: _NumpyArray_OfNumber,
        *args: *_Any_Number_Ts,
        entry: Optional[_NumpyArray_OfNumber] = None,
        optimizer_options: Optional[dict[str, Any]] = None,
    ) -> None: ...