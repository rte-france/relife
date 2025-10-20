from abc import ABC
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Self,
    Union,
    overload,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds
from typing_extensions import override

from relife._typing import (
    _Any_Integer,
    _Any_Number,
    _Any_Numpy_Bool,
    _Any_Numpy_Number,
    _NumpyArray_OfBool,
    _NumpyArray_OfNumber,
)

from ._base import FittableParametricLifetimeModel, ParametricLifetimeModel

__all__ = [
    "LifetimeDistribution",
    "Gompertz",
    "Weibull",
    "Gamma",
    "LogLogistic",
    "EquilibriumDistribution",
    "Exponential",
    "MinimumDistribution",
]

class LifetimeDistribution(FittableParametricLifetimeModel[()], ABC):
    def sf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    def pdf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def isf(self, probability: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def cdf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def ppf(self, probability: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def median(self) -> np.float64: ...
    @override
    def moment(self, n: int) -> np.float64: ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_Any_Number], _Any_Numpy_Number],
        a: _Any_Number,
        b: _Any_Number,
        deg: int = 10,
    ) -> _Any_Numpy_Number: ...
    @overload
    def jac_sf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[False],
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_sf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[True],
    ) -> _Any_Numpy_Number: ...
    def jac_sf(
        self,
        time: _Any_Number,
        *,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def jac_cdf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[False],
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_cdf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[True],
    ) -> _Any_Numpy_Number: ...
    def jac_cdf(
        self,
        time: _Any_Number,
        *,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def jac_pdf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[False],
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_pdf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[True],
    ) -> _Any_Numpy_Number: ...
    def jac_pdf(
        self,
        time: _Any_Number,
        *,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
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
        *,
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
        *,
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
        *,
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
        *,
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
    def _get_initial_params(
        self,
        time: _NumpyArray_OfNumber,
        event: Optional[_NumpyArray_OfBool] = None,
        entry: Optional[_NumpyArray_OfNumber] = None,
    ) -> NDArray[np.float64]: ...
    def _get_params_bounds(self) -> Bounds: ...
    @override
    def fit(
        self,
        time: _NumpyArray_OfNumber,
        event: Optional[_NumpyArray_OfBool] = None,
        entry: Optional[_NumpyArray_OfNumber] = None,
        optimizer_options: Optional[dict[str, Any]] = None,
    ) -> Self: ...
    @override
    def fit_from_interval_censored_lifetimes(
        self,
        time_inf: _NumpyArray_OfNumber,
        time_sup: _NumpyArray_OfNumber,
        entry: Optional[_NumpyArray_OfNumber] = None,
        optimizer_options: Optional[dict[str, Any]] = None,
    ) -> Self: ...

class Exponential(LifetimeDistribution):
    def __init__(self, rate: Optional[float] = None) -> None: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    def chf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def mrl(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def ichf(self, cumulative_hazard_rate: _Any_Number) -> _Any_Numpy_Number: ...
    def dhf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @overload
    def jac_hf(
        self, time: _Any_Number, *, asarray: Literal[False]
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_hf(
        self, time: _Any_Number, *, asarray: Literal[True]
    ) -> _Any_Numpy_Number: ...
    def jac_hf(
        self,
        time: _Any_Number,
        *,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def jac_chf(
        self, time: _Any_Number, *, asarray: Literal[False]
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_chf(
        self, time: _Any_Number, *, asarray: Literal[True]
    ) -> _Any_Numpy_Number: ...
    def jac_chf(
        self, time: _Any_Number, *, asarray: bool = True
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...

class Weibull(LifetimeDistribution):
    def __init__(
        self, shape: Optional[float] = None, rate: Optional[float] = None
    ) -> None: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    def chf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def mrl(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def ichf(self, cumulative_hazard_rate: _Any_Number) -> _Any_Numpy_Number: ...
    def dhf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @overload
    def jac_hf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[False],
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[True],
    ) -> _Any_Numpy_Number: ...
    def jac_hf(
        self,
        time: _Any_Number,
        *,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def jac_chf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[False],
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[True],
    ) -> _Any_Numpy_Number: ...
    def jac_chf(
        self,
        time: _Any_Number,
        *,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...

class Gompertz(LifetimeDistribution):
    def __init__(
        self, shape: float | None = None, rate: float | None = None
    ) -> None: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    def chf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def mrl(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def ichf(self, cumulative_hazard_rate: _Any_Number) -> _Any_Numpy_Number: ...
    def dhf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @overload
    def jac_hf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[False],
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[True],
    ) -> _Any_Numpy_Number: ...
    def jac_hf(
        self,
        time: _Any_Number,
        *,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def jac_chf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[False],
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[True],
    ) -> _Any_Numpy_Number: ...
    def jac_chf(
        self,
        time: _Any_Number,
        *,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...

class Gamma(LifetimeDistribution):
    def __init__(
        self, shape: float | None = None, rate: float | None = None
    ) -> None: ...
    def _uppergamma(self, x: _Any_Number) -> _Any_Numpy_Number: ...
    def _jac_uppergamma_shape(self, x: _Any_Number) -> _Any_Numpy_Number: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    def chf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def ichf(self, cumulative_hazard_rate: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def mrl(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    def dhf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @overload
    def jac_hf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[False],
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[True],
    ) -> _Any_Numpy_Number: ...
    def jac_hf(
        self,
        time: _Any_Number,
        *,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def jac_chf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[False],
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _Any_Number,
        *,
        asarray: Literal[True],
    ) -> _Any_Numpy_Number: ...
    def jac_chf(
        self,
        time: _Any_Number,
        *,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...

class LogLogistic(LifetimeDistribution):
    def __init__(
        self, shape: float | None = None, rate: float | None = None
    ) -> None: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    def chf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def ichf(self, cumulative_hazard_rate: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def mrl(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    def dhf(self, time: _Any_Number) -> _Any_Numpy_Number: ...
    @overload
    def jac_hf(
        self, time: _Any_Number, *, asarray: Literal[False]
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_hf(
        self, time: _Any_Number, *, asarray: Literal[True]
    ) -> _Any_Numpy_Number: ...
    def jac_hf(
        self, time: _Any_Number, *, asarray: bool = True
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def jac_chf(
        self, time: _Any_Number, *, asarray: Literal[False]
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_chf(
        self, time: _Any_Number, *, asarray: Literal[True]
    ) -> _Any_Numpy_Number: ...
    def jac_chf(
        self, time: _Any_Number, *, asarray: bool = True
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...

class EquilibriumDistribution(ParametricLifetimeModel[*tuple[_Any_Number, ...]]):
    baseline: ParametricLifetimeModel[*tuple[_Any_Number, ...]]
    def __init__(
        self, baseline: ParametricLifetimeModel[*tuple[_Any_Number, ...]]
    ) -> None: ...
    @override
    def cdf(self, time: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    def sf(self, time: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    def pdf(self, time: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    def hf(self, time: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    def chf(self, time: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def isf(
        self, probability: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def ichf(
        self, cumulative_hazard_rate: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...

class MinimumDistribution(
    FittableParametricLifetimeModel[_Any_Integer, *tuple[_Any_Number, ...]]
):
    baseline: FittableParametricLifetimeModel[*tuple[_Any_Number, ...]]
    def __init__(
        self, baseline: FittableParametricLifetimeModel[*tuple[_Any_Number, ...]]
    ) -> None: ...
    @override
    def sf(
        self, time: _Any_Number, n: _Any_Integer, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def pdf(
        self, time: _Any_Number, n: _Any_Integer, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def hf(
        self, time: _Any_Number, n: _Any_Integer, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def chf(
        self, time: _Any_Number, n: _Any_Integer, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def ichf(
        self,
        cumulative_hazard_rate: _Any_Number,
        n: _Any_Integer,
        *args: _Any_Number,
    ) -> _Any_Numpy_Number: ...
    def dhf(
        self, time: _Any_Number, n: _Any_Integer, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    def jac_chf(
        self,
        time: _Any_Number,
        n: _Any_Integer,
        *args: _Any_Number,
        asarray: bool = False,
    ) -> _Any_Numpy_Number | tuple[_Any_Numpy_Number, ...]: ...
    def jac_hf(
        self,
        time: _Any_Number,
        n: _Any_Integer,
        *args: _Any_Number,
        asarray: bool = False,
    ) -> _Any_Numpy_Number | tuple[_Any_Numpy_Number, ...]: ...
    def jac_sf(
        self,
        time: _Any_Number,
        n: _Any_Integer,
        *args: _Any_Number,
        asarray: bool = False,
    ) -> _Any_Numpy_Number | tuple[_Any_Numpy_Number, ...]: ...
    def jac_cdf(
        self,
        time: _Any_Number,
        n: _Any_Integer,
        *args: _Any_Number,
        asarray: bool = False,
    ) -> _Any_Numpy_Number | tuple[_Any_Numpy_Number, ...]: ...
    def jac_pdf(
        self,
        time: _Any_Number,
        n: _Any_Integer,
        *args: _Any_Number,
        asarray: bool = False,
    ) -> _Any_Numpy_Number | tuple[_Any_Numpy_Number, ...]: ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_Any_Number], _Any_Numpy_Number],
        a: _Any_Number,
        b: _Any_Number,
        n: _Any_Integer,
        *args: _Any_Number,
        deg: int = 10,
    ) -> _Any_Numpy_Number: ...
    def _get_initial_params(
        self,
        time: _NumpyArray_OfNumber,
        n: _Any_Integer,
        *args: _Any_Number,
        event: Optional[_NumpyArray_OfBool] = None,
        entry: Optional[_NumpyArray_OfNumber] = None,
    ) -> NDArray[np.float64]: ...
    def _get_params_bounds(self) -> Bounds: ...
    @override
    def fit(
        self,
        time: _NumpyArray_OfNumber,
        n: _Any_Integer,
        *args: _Any_Number,
        event: Optional[_NumpyArray_OfBool] = None,
        entry: Optional[_NumpyArray_OfNumber] = None,
        optimizer_options: Optional[dict[str, Any]] = None,
    ) -> Self: ...
    @override
    def fit_from_interval_censored_lifetimes(
        self,
        time_inf: _NumpyArray_OfNumber,
        time_sup: _NumpyArray_OfNumber,
        n: _Any_Integer,
        *args: _Any_Number,
        entry: Optional[_NumpyArray_OfNumber] = None,
        optimizer_options: Optional[dict[str, Any]] = None,
    ) -> Self: ...
