from abc import ABC
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Self,
    overload, Union,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds
from typing_extensions import override

from relife import ParametricModel as ParametricModel
from relife._typing import _B, _X, _Y, _Xs
from relife.lifetime_model import (
    FittableParametricLifetimeModel,
    FrozenParametricLifetimeModel,
)
from relife.likelihood import FittingResults as FittingResults

def broadcast_time_covar(time: _X, covar: _X) -> tuple[_X, _X]: ...
def broadcast_time_covar_shapes(
    time_shape: tuple[()] | tuple[int] | tuple[int, int], covar_shape: tuple[()] | tuple[int] | tuple[int, int]
) -> tuple[()] | tuple[int] | tuple[int, int]: ...

class LifetimeRegression(FittableParametricLifetimeModel[*tuple[_X, *_Xs]], ABC):
    fitting_results = Optional[FittingResults]
    covar_effect: CovarEffect
    baseline: FittableParametricLifetimeModel[*_Xs]

    def __init__(
        self,
        baseline: FittableParametricLifetimeModel[*_Xs],
        coefficients: tuple[Optional[float], ...] = (None,),
    ) -> None: ...
    def _get_initial_params(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *_Xs,
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
    def sf(self, time: _X, covar: _X, *args: *_Xs) -> _Y: ...
    @override
    def pdf(self, time: _X, covar: _X, *args: *_Xs) -> _Y: ...
    @override
    def isf(
        self,
        probability: _X,
        covar: _X,
        *args: *_Xs,
    ) -> _Y: ...
    @override
    def cdf(self, time: _X, covar: _X, *args: *_Xs) -> _Y: ...
    @override
    def ppf(
        self,
        probability: _X,
        covar: _X,
        *args: *_Xs,
    ) -> _Y: ...
    @override
    def mrl(self, time: _X, covar: _X, *args: *_Xs) -> _Y: ...
    @override
    def mean(self, covar: _X, *args: *_Xs) -> _Y: ...
    @override
    def var(self, covar: _X, *args: *_Xs) -> _Y: ...
    @override
    def median(self, covar: _X, *args: *_Xs) -> _Y: ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_X], _Y],
        a: _X,
        b: _X,
        covar: _X,
        *args: *_Xs,
        deg: int = 10,
    ) -> _Y: ...
    @overload
    def jac_sf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: Literal[False] = False,
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_sf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: Literal[True] = True,
    ) -> _Y: ...
    def jac_sf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_cdf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: Literal[False] = False,
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_cdf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: Literal[True] = True,
    ) -> _Y: ...
    def jac_cdf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_pdf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: Literal[False] = False,
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_pdf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: Literal[True] = True,
    ) -> _Y: ...
    def jac_pdf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def rvs(
        self,
        size: int,
        covar: _X,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> _Y: ...
    @overload
    def rvs(
        self,
        size: int,
        covar: _X,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> tuple[_Y, _B]: ...
    @overload
    def rvs(
        self,
        size: int,
        covar: _X,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> tuple[_Y, _Y]: ...
    @overload
    def rvs(
        self,
        size: int,
        covar: _X,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> tuple[_Y, _B, _Y]: ...
    def rvs(
        self,
        size: int,
        covar: _X,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> _Y | tuple[_Y, _Y] | tuple[_Y, _B] | tuple[_Y, _B, _Y]: ...
    def fit(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
        departure: NDArray[np.float64] | None = None,
        **options: Any,
    ) -> Self: ...
    def freeze(self, covar: _X, *args: *_Xs) -> FrozenLifetimeRegression[*_Xs]: ...

class ProportionalHazard(LifetimeRegression[*_Xs]):
    def hf(self, time: _X, covar: _X, *args: *_Xs) -> _Y: ...
    def chf(self, time: _X, covar: _X, *args: *_Xs) -> _Y: ...
    @override
    def ichf(
        self,
        cumulative_hazard_rate: _X,
        covar: _X,
        *args: *_Xs,
    ) -> _Y: ...
    def dhf(self, time: _X, covar: _X, *args: *_Xs) -> _Y: ...
    @overload
    def jac_hf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: Literal[False] = False,
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: Literal[True] = True,
    ) -> _Y: ...
    def jac_hf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_chf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: Literal[False] = False,
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: Literal[True] = True,
    ) -> _Y: ...
    def jac_chf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @override
    def moment(self, n: int, covar: _X, *args: *_Xs) -> _Y: ...

class AcceleratedFailureTime(LifetimeRegression[*_Xs]):
    def hf(self, time: _X, covar: _X, *args: *_Xs) -> _Y: ...
    def chf(self, time: _X, covar: _X, *args: *_Xs) -> _Y: ...
    @override
    def ichf(
        self,
        cumulative_hazard_rate: _X,
        covar: _X,
        *args: *_Xs,
    ) -> _Y: ...
    def dhf(self, time: _X, covar: _X, *args: *_Xs) -> _Y: ...
    @overload
    def jac_hf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: Literal[False] = False,
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: Literal[True] = True,
    ) -> _Y: ...
    def jac_hf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_chf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: Literal[False] = False,
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: Literal[True] = True,
    ) -> _Y: ...
    def jac_chf(
        self,
        time: _X,
        covar: _X,
        *args: *_Xs,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @override
    def moment(self, n: int, covar: _X, *args: *_Xs) -> _Y: ...

class CovarEffect(ParametricModel):
    def __init__(self, coefficients: tuple[Optional[None], ...] = (None,)) -> None: ...
    @property
    def nb_coef(self) -> int: ...
    def g(self, covar: _X) -> _Y: ...
    @overload
    def jac_g(self, covar: _X, *, asarray: Literal[False] = False) -> tuple[_Y, ...]: ...
    @overload
    def jac_g(self, covar: _X, *, asarray: Literal[True] = True) -> _Y: ...
    def jac_g(self, covar: _X, *, asarray: Literal[True] = True) -> tuple[_Y, ...] | _Y: ...

class FrozenLifetimeRegression(FrozenParametricLifetimeModel[*tuple[_X, *_Xs]]):
    unfrozen_model: LifetimeRegression[*_Xs]
    args: tuple[_X, *_Xs]

    def __init__(self, model: LifetimeRegression[*_Xs], covar: _X, *args: *_Xs) -> None: ...
    @override
    def unfreeze(self) -> LifetimeRegression[*_Xs]: ...
    @property
    def nb_coef(self) -> int: ...
    @property
    def covar(self) -> _X: ...
    # noinspection PyUnresolvedReferences
    @covar.setter
    def covar(self, value: _X) -> None: ...
    def dhf(self, time: _X, *args: *_Xs) -> _Y: ...
    @overload
    def jac_hf(self, time: _X, asarray: Literal[False] = False) -> tuple[_Y, ...]: ...
    @overload
    def jac_hf(self, time: _X, asarray: Literal[True] = True) -> _Y: ...
    def jac_hf(self, time: _X, asarray: bool = True) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_chf(self, time: _X, asarray: Literal[False] = False) -> tuple[_Y, ...]: ...
    @overload
    def jac_chf(self, time: _X, asarray: Literal[True] = True) -> _Y: ...
    def jac_chf(self, time: _X, asarray: bool = True) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_sf(self, time: _X, asarray: Literal[False] = False) -> tuple[_Y, ...]: ...
    @overload
    def jac_sf(self, time: _X, asarray: Literal[True] = True) -> _Y: ...
    def jac_sf(self, time: _X, asarray: bool = True) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_pdf(self, time: _X, asarray: Literal[False] = False) -> tuple[_Y, ...]: ...
    @overload
    def jac_pdf(self, time: _X, asarray: Literal[True] = True) -> _Y: ...
    def jac_pdf(self, time: _X, asarray: bool = True) -> tuple[_Y, ...] | _Y: ...
