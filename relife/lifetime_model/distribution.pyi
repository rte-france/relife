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

from relife._io_types import (
    _AdditionalIntOrFloatValues,
    _BooleanValues,
    _IntOrFloatValues,
    _NumpyFloatValues,
)
from relife.lifetime_model import LifetimeRegression
from relife.likelihood import FittingResults

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
    fitting_results: Optional[FittingResults]

    def sf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def pdf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def isf(self, probability: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def cdf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def ppf(self, probability: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def median(self) -> np.float64: ...
    @override
    def moment(self, n: int) -> np.float64: ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_IntOrFloatValues], _NumpyFloatValues],
        a: _IntOrFloatValues,
        b: _IntOrFloatValues,
        deg: int = 10,
    ) -> _NumpyFloatValues: ...
    @overload
    def jac_sf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[False],
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_sf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[True],
    ) -> _NumpyFloatValues: ...
    def jac_sf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def jac_cdf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[False],
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_cdf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[True],
    ) -> _NumpyFloatValues: ...
    def jac_cdf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def jac_pdf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[False],
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_pdf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[True],
    ) -> _NumpyFloatValues: ...
    def jac_pdf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> _NumpyFloatValues: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _BooleanValues]: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _NumpyFloatValues]: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _BooleanValues, _NumpyFloatValues]: ...
    def rvs(
        self,
        size: int,
        *,
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
    def _get_initial_params(self, time, event=None, entry=None, departure=None) -> NDArray[np.float64]: ...
    def _get_params_bounds(self) -> Bounds: ...
    @override
    def fit(
        self,
        time: NDArray[np.float64],
        *,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        **options: Any,
    ) -> Self: ...

class Exponential(LifetimeDistribution):
    def __init__(self, rate: Optional[float] = None) -> None: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def chf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def mrl(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def ichf(self, cumulative_hazard_rate: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def dhf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @overload
    def jac_hf(self, time: _IntOrFloatValues, *, asarray: Literal[False]) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_hf(self, time: _IntOrFloatValues, *, asarray: Literal[True]) -> _NumpyFloatValues: ...
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def jac_chf(self, time: _IntOrFloatValues, *, asarray: Literal[False]) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_chf(self, time: _IntOrFloatValues, *, asarray: Literal[True]) -> _NumpyFloatValues: ...
    def jac_chf(
        self, time: _IntOrFloatValues, *, asarray: bool = True
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...

class Weibull(LifetimeDistribution):
    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None) -> None: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def chf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def mrl(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def ichf(self, cumulative_hazard_rate: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def dhf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @overload
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[False],
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[True],
    ) -> _NumpyFloatValues: ...
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[False],
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[True],
    ) -> _NumpyFloatValues: ...
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...

class Gompertz(LifetimeDistribution):
    def __init__(self, shape: float | None = None, rate: float | None = None) -> None: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def chf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def mrl(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def ichf(self, cumulative_hazard_rate: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def dhf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @overload
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[False],
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[True],
    ) -> _NumpyFloatValues: ...
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[False],
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[True],
    ) -> _NumpyFloatValues: ...
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...

class Gamma(LifetimeDistribution):
    def __init__(self, shape: float | None = None, rate: float | None = None) -> None: ...
    def _uppergamma(self, x: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def _jac_uppergamma_shape(self, x: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def chf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def ichf(self, cumulative_hazard_rate: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def mrl(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def dhf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @overload
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[False],
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[True],
    ) -> _NumpyFloatValues: ...
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[False],
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: Literal[True],
    ) -> _NumpyFloatValues: ...
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        *,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...

class LogLogistic(LifetimeDistribution):
    def __init__(self, shape: float | None = None, rate: float | None = None) -> None: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def chf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def ichf(self, cumulative_hazard_rate: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def mrl(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def dhf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @overload
    def jac_hf(self, time: _IntOrFloatValues, *, asarray: Literal[False]) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_hf(self, time: _IntOrFloatValues, *, asarray: Literal[True]) -> _NumpyFloatValues: ...
    def jac_hf(
        self, time: _IntOrFloatValues, *, asarray: bool = True
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def jac_chf(self, time: _IntOrFloatValues, *, asarray: Literal[False]) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_chf(self, time: _IntOrFloatValues, *, asarray: Literal[True]) -> _NumpyFloatValues: ...
    def jac_chf(
        self, time: _IntOrFloatValues, *, asarray: bool = True
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...

class EquilibriumDistribution(ParametricLifetimeModel[*tuple[_IntOrFloatValues, *_AdditionalIntOrFloatValues]]):
    baseline: ParametricLifetimeModel[*tuple[_IntOrFloatValues, *_AdditionalIntOrFloatValues]]
    fitting_results = Optional[FittingResults]
    def __init__(
        self, baseline: ParametricLifetimeModel[*tuple[_IntOrFloatValues, *_AdditionalIntOrFloatValues]]
    ) -> None: ...
    @property
    def args_names(self) -> tuple[str, ...]: ...
    @override
    def cdf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def sf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def pdf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def hf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def chf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def isf(self, probability: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def ichf(
        self, cumulative_hazard_rate: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...

class MinimumDistribution(ParametricLifetimeModel[*tuple[int | NDArray[np.int64], *_AdditionalIntOrFloatValues]]):
    baseline: FittableParametricLifetimeModel[*_AdditionalIntOrFloatValues]
    fitting_results = Optional[FittingResults]
    def __init__(self, baseline: LifetimeDistribution | LifetimeRegression) -> None: ...
    @override
    def sf(
        self, time: _IntOrFloatValues, n: int | NDArray[np.int64], *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def pdf(
        self, time: _IntOrFloatValues, n: int | NDArray[np.int64], *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def hf(
        self, time: _IntOrFloatValues, n: int | NDArray[np.int64], *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def chf(
        self, time: _IntOrFloatValues, n: int | NDArray[np.int64], *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def ichf(
        self,
        cumulative_hazard_rate: _IntOrFloatValues,
        n: int | NDArray[np.int64],
        *args: *_AdditionalIntOrFloatValues,
    ) -> _NumpyFloatValues: ...
    def dhf(
        self, time: _IntOrFloatValues, n: int | NDArray[np.int64], *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        n: int | NDArray[np.int64],
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = False,
    ) -> _NumpyFloatValues | tuple[_NumpyFloatValues, ...]: ...
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        n: int | NDArray[np.int64],
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = False,
    ) -> _NumpyFloatValues | tuple[_NumpyFloatValues, ...]: ...
    def jac_sf(
        self,
        time: _IntOrFloatValues,
        n: int | NDArray[np.int64],
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = False,
    ) -> _NumpyFloatValues | tuple[_NumpyFloatValues, ...]: ...
    def jac_cdf(
        self,
        time: _IntOrFloatValues,
        n: int | NDArray[np.int64],
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = False,
    ) -> _NumpyFloatValues | tuple[_NumpyFloatValues, ...]: ...
    def jac_pdf(
        self,
        time: _IntOrFloatValues,
        n: int | NDArray[np.int64],
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = False,
    ) -> _NumpyFloatValues | tuple[_NumpyFloatValues, ...]: ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_IntOrFloatValues], _NumpyFloatValues],
        a: _IntOrFloatValues,
        b: _IntOrFloatValues,
        n: int | NDArray[np.int64],
        *args: *_AdditionalIntOrFloatValues,
        deg: int = 10,
    ) -> _NumpyFloatValues: ...
    def fit(
        self,
        time: NDArray[np.float64],
        n: NDArray[np.int64],
        *args: NDArray[np.float64],
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> Self: ...
