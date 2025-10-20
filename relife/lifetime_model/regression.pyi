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
    _Any_Number,
    _Any_Numpy_Bool,
    _Any_Numpy_Number,
    _NumpyArray_OfBool,
    _NumpyArray_OfNumber,
)
from relife.base import FrozenParametricModel, ParametricModel

from .distribution import LifetimeDistribution
from ._base import FittableParametricLifetimeModel

__all__ = ["LifetimeRegression", "AcceleratedFailureTime", "ProportionalHazard"]

def _broadcast_time_covar(
    time: _Any_Number, covar: _Any_Number
) -> tuple[_Any_Numpy_Number, _Any_Numpy_Number]: ...
def _broadcast_time_covar_shapes(
    time_shape: tuple[()] | tuple[int] | tuple[int, int],
    covar_shape: tuple[()] | tuple[int] | tuple[int, int],
) -> tuple[()] | tuple[int] | tuple[int, int]: ...


class LifetimeRegression(FittableParametricLifetimeModel[_Any_Number], ABC):
    covar_effect: _CovarEffect
    baseline: LifetimeDistribution

    def __init__(self, baseline: LifetimeDistribution, coefficients: tuple[Optional[float], ...] = (None,),) -> None: ...
    @property
    def coefficients(self) -> NDArray[np.float64]: ...
    @property
    def nb_coef(self) -> int: ...
    @override
    def sf(self, time: _Any_Number, covar: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def pdf(self, time: _Any_Number, covar: _Any_Number) -> _Any_Numpy_Number: ...
    def isf(self, probability: _Any_Number, covar: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def cdf(self, time: _Any_Number, covar: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def ppf(self, probability: _Any_Number, covar: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def mrl(self, time: _Any_Number, covar: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def mean(self, covar: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def var(self, covar: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def median(self, covar: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_Any_Number], _Any_Numpy_Number],
        a: _Any_Number,
        b: _Any_Number,
        covar: _Any_Number,
        deg: int = 10,
    ) -> _Any_Numpy_Number: ...
    @overload
    def jac_sf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: Literal[False] = False,
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_sf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: Literal[True] = True,
    ) -> _Any_Numpy_Number: ...
    def jac_sf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def jac_cdf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: Literal[False] = False,
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_cdf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: Literal[True] = True,
    ) -> _Any_Numpy_Number: ...
    def jac_cdf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def jac_pdf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: Literal[False] = False,
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_pdf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: Literal[True] = True,
    ) -> _Any_Numpy_Number: ...
    def jac_pdf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def rvs(
        self,
        size: int,
        covar: _Any_Number,
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
        covar: _Any_Number,
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
        covar: _Any_Number,
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
        covar: _Any_Number,
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
        covar: _Any_Number,
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
    def _get_params_bounds(self) -> Bounds: ...
    def _get_initial_params(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]: ...
    @override
    def fit(
        self,
        time: _NumpyArray_OfNumber,
        covar: _Any_Number,
        event: Optional[_NumpyArray_OfBool] = None,
        entry: Optional[_NumpyArray_OfNumber] = None,
        optimizer_options: Optional[dict[str, Any]] = None,
    ) -> Self: ...
    @override
    def fit_from_interval_censored_lifetimes(
        self,
        time: _NumpyArray_OfNumber,
        covar: _Any_Number,
        event: Optional[_NumpyArray_OfBool] = None,
        entry: Optional[_NumpyArray_OfNumber] = None,
        optimizer_options: Optional[dict[str, Any]] = None,
    ) -> Self: ...
    def freeze(
        self, covar: _Any_Number
    ) -> FrozenParametricModel[
        FittableParametricLifetimeModel[_Any_Number]
    ]: ...

class ProportionalHazard(LifetimeRegression):
    def hf(self, time: _Any_Number, covar: _Any_Number) -> _Any_Numpy_Number: ...
    def chf(self, time: _Any_Number, covar: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def ichf(self, cumulative_hazard_rate: _Any_Number, covar: _Any_Number) -> _Any_Numpy_Number: ...
    def dhf(self, time: _Any_Number, covar: _Any_Number) -> _Any_Numpy_Number: ...
    @overload
    def jac_hf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: Literal[False] = False,
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: Literal[True] = True,
    ) -> _Any_Numpy_Number: ...
    def jac_hf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def jac_chf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: Literal[False] = False,
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: Literal[True] = True,
    ) -> _Any_Numpy_Number: ...
    def jac_chf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @override
    def moment(self, n: int, covar: _Any_Number) -> _Any_Numpy_Number: ...

class AcceleratedFailureTime(LifetimeRegression):
    def hf(self, time: _Any_Number, covar: _Any_Number) -> _Any_Numpy_Number: ...
    def chf(self, time: _Any_Number, covar: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def ichf(self, cumulative_hazard_rate: _Any_Number, covar: _Any_Number) -> _Any_Numpy_Number: ...
    def dhf(self, time: _Any_Number, covar: _Any_Number) -> _Any_Numpy_Number: ...
    @overload
    def jac_hf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: Literal[False] = False,
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: Literal[True] = True,
    ) -> _Any_Numpy_Number: ...
    def jac_hf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @overload
    def jac_chf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: Literal[False] = False,
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: Literal[True] = True,
    ) -> _Any_Numpy_Number: ...
    def jac_chf(
        self,
        time: _Any_Number,
        covar: _Any_Number,
        asarray: bool = True,
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
    @override
    def moment(self, n: int, covar: _Any_Number) -> _Any_Numpy_Number: ...

class _CovarEffect(ParametricModel):
    def __init__(self, coefficients: tuple[Optional[None], ...] = (None,)) -> None: ...
    @property
    def nb_coef(self) -> int: ...
    def g(self, covar: _Any_Number) -> _Any_Numpy_Number: ...
    @overload
    def jac_g(
        self, covar: _Any_Number, asarray: Literal[False] = False
    ) -> tuple[_Any_Numpy_Number, ...]: ...
    @overload
    def jac_g(
        self, covar: _Any_Number, asarray: Literal[True] = True
    ) -> _Any_Numpy_Number: ...
    def jac_g(
        self, covar: _Any_Number, asarray: Literal[True] = True
    ) -> tuple[_Any_Numpy_Number, ...] | _Any_Numpy_Number: ...
