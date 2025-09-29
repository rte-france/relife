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

from relife import ParametricModel
from relife._typing import (
    _AdditionalIntOrFloatValues,
    _BooleanValues,
    _IntOrFloatValues,
    _NumpyFloatValues,
)
from relife.lifetime_model import FittableParametricLifetimeModel

from ._typing import _FrozenParametricLifetimeModel

def _broadcast_time_covar(
    time: _IntOrFloatValues, covar: _IntOrFloatValues
) -> tuple[_IntOrFloatValues, _IntOrFloatValues]: ...
def _broadcast_time_covar_shapes(
    time_shape: tuple[()] | tuple[int] | tuple[int, int], covar_shape: tuple[()] | tuple[int] | tuple[int, int]
) -> tuple[()] | tuple[int] | tuple[int, int]: ...

class LifetimeRegression(FittableParametricLifetimeModel[*tuple[_IntOrFloatValues, *_AdditionalIntOrFloatValues]], ABC):
    covar_effect: CovarEffect
    baseline: FittableParametricLifetimeModel[*_AdditionalIntOrFloatValues]

    def __init__(
        self,
        baseline: FittableParametricLifetimeModel[*_AdditionalIntOrFloatValues],
        coefficients: tuple[Optional[float], ...] = (None,),
    ) -> None: ...
    def _get_initial_params(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *_AdditionalIntOrFloatValues,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]: ...
    def _get_params_bounds(self) -> Bounds: ...
    @property
    def coefficients(self) -> NDArray[np.float64]: ...
    @property
    def nb_coef(self) -> int: ...
    @override
    def sf(
        self, time: _IntOrFloatValues, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def pdf(
        self, time: _IntOrFloatValues, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def isf(
        self,
        probability: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
    ) -> _NumpyFloatValues: ...
    @override
    def cdf(
        self, time: _IntOrFloatValues, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def ppf(
        self,
        probability: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
    ) -> _NumpyFloatValues: ...
    @override
    def mrl(
        self, time: _IntOrFloatValues, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def mean(self, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def var(self, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def median(self, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_IntOrFloatValues], _NumpyFloatValues],
        a: _IntOrFloatValues,
        b: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        deg: int = 10,
    ) -> _NumpyFloatValues: ...
    @overload
    def jac_sf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[False] = False,
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_sf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[True] = True,
    ) -> _NumpyFloatValues: ...
    def jac_sf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def jac_cdf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[False] = False,
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_cdf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[True] = True,
    ) -> _NumpyFloatValues: ...
    def jac_cdf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def jac_pdf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[False] = False,
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_pdf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[True] = True,
    ) -> _NumpyFloatValues: ...
    def jac_pdf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def rvs(
        self,
        size: int,
        covar: _IntOrFloatValues,
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
        covar: _IntOrFloatValues,
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
        covar: _IntOrFloatValues,
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
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _BooleanValues, _NumpyFloatValues]: ...
    def rvs(
        self,
        size: int,
        covar: _IntOrFloatValues,
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
    def fit(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        **options: Any,
    ) -> Self: ...
    def freeze(
        self, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues
    ) -> _FrozenParametricLifetimeModel[*tuple[_IntOrFloatValues, _AdditionalIntOrFloatValues]]: ...

class ProportionalHazard(LifetimeRegression[*_AdditionalIntOrFloatValues]):
    def hf(
        self, time: _IntOrFloatValues, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    def chf(
        self, time: _IntOrFloatValues, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def ichf(
        self,
        cumulative_hazard_rate: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
    ) -> _NumpyFloatValues: ...
    def dhf(
        self, time: _IntOrFloatValues, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @overload
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[False] = False,
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[True] = True,
    ) -> _NumpyFloatValues: ...
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[False] = False,
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[True] = True,
    ) -> _NumpyFloatValues: ...
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @override
    def moment(self, n: int, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...

class AcceleratedFailureTime(LifetimeRegression[*_AdditionalIntOrFloatValues]):
    def hf(
        self, time: _IntOrFloatValues, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    def chf(
        self, time: _IntOrFloatValues, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def ichf(
        self,
        cumulative_hazard_rate: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
    ) -> _NumpyFloatValues: ...
    def dhf(
        self, time: _IntOrFloatValues, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @overload
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[False] = False,
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[True] = True,
    ) -> _NumpyFloatValues: ...
    def jac_hf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @overload
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[False] = False,
    ) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: Literal[True] = True,
    ) -> _NumpyFloatValues: ...
    def jac_chf(
        self,
        time: _IntOrFloatValues,
        covar: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        asarray: bool = True,
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
    @override
    def moment(self, n: int, covar: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...

class CovarEffect(ParametricModel):
    def __init__(self, coefficients: tuple[Optional[None], ...] = (None,)) -> None: ...
    @property
    def nb_coef(self) -> int: ...
    def g(self, covar: _IntOrFloatValues) -> _NumpyFloatValues: ...
    @overload
    def jac_g(self, covar: _IntOrFloatValues, *, asarray: Literal[False] = False) -> tuple[_NumpyFloatValues, ...]: ...
    @overload
    def jac_g(self, covar: _IntOrFloatValues, *, asarray: Literal[True] = True) -> _NumpyFloatValues: ...
    def jac_g(
        self, covar: _IntOrFloatValues, *, asarray: Literal[True] = True
    ) -> tuple[_NumpyFloatValues, ...] | _NumpyFloatValues: ...
