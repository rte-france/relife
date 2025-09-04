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

from relife._typing import _B, _X, _Y, _Xs
from relife.lifetime_model import LifetimeRegression
from relife.likelihood import FittingResults

from ._base import FittableParametricLifetimeModel, ParametricLifetimeModel

class LifetimeDistribution(FittableParametricLifetimeModel[()], ABC):
    fitting_results: Optional[FittingResults]

    def sf(self, time: _X) -> _Y: ...
    def pdf(self, time: _X) -> _Y: ...
    @override
    def isf(self, probability: _X) -> _Y: ...
    @override
    def cdf(self, time: _X) -> _Y: ...
    @override
    def ppf(self, probability: _X) -> _Y: ...
    @override
    def median(self) -> np.float64: ...
    @override
    def moment(self, n: int) -> np.float64: ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_X], _Y],
        a: _X,
        b: _X,
        deg: int = 10,
    ) -> _Y: ...
    @overload
    def jac_sf(
        self,
        time: _X,
        *,
        asarray: Literal[False],
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_sf(
        self,
        time: _X,
        *,
        asarray: Literal[True],
    ) -> _Y: ...
    def jac_sf(
        self,
        time: _X,
        *,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_cdf(
        self,
        time: _X,
        *,
        asarray: Literal[False],
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_cdf(
        self,
        time: _X,
        *,
        asarray: Literal[True],
    ) -> _Y: ...
    def jac_cdf(
        self,
        time: _X,
        *,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_pdf(
        self,
        time: _X,
        *,
        asarray: Literal[False],
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_pdf(
        self,
        time: _X,
        *,
        asarray: Literal[True],
    ) -> _Y: ...
    def jac_pdf(
        self,
        time: _X,
        *,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> _Y: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> tuple[_Y, _B]: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> tuple[_Y, _Y]: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> tuple[_Y, _B, _Y]: ...
    def rvs(
        self,
        size: int,
        *,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> _Y | tuple[_Y, _Y] | tuple[_Y, _B] | tuple[_Y, _B, _Y]: ...
    def _get_initial_params(self, time, event=None, entry=None, departure=None) -> NDArray[np.float64]: ...
    def _get_params_bounds(self) -> Bounds: ...
    @override
    def fit(
        self,
        time: NDArray[np.float64],
        *,
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
        departure: NDArray[np.float64] | None = None,
        **options: Any,
    ) -> Self: ...

class Exponential(LifetimeDistribution):
    def __init__(self, rate: Optional[float] = None) -> None: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _X) -> _Y: ...
    def chf(self, time: _X) -> _Y: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def mrl(self, time: _X) -> _Y: ...
    @override
    def ichf(self, cumulative_hazard_rate: _X) -> _Y: ...
    def dhf(self, time: _X) -> _Y: ...
    @overload
    def jac_hf(self, time: _X, *, asarray: Literal[False]) -> tuple[_Y, ...]: ...
    @overload
    def jac_hf(self, time: _X, *, asarray: Literal[True]) -> _Y: ...
    def jac_hf(
        self,
        time: _X,
        *,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_chf(self, time: _X, *, asarray: Literal[False]) -> tuple[_Y, ...]: ...
    @overload
    def jac_chf(self, time: _X, *, asarray: Literal[True]) -> _Y: ...
    def jac_chf(self, time: _X, *, asarray: bool = True) -> tuple[_Y, ...] | _Y: ...

class Weibull(LifetimeDistribution):
    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None) -> None: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _X) -> _Y: ...
    def chf(self, time: _X) -> _Y: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def mrl(self, time: _X) -> _Y: ...
    @override
    def ichf(self, cumulative_hazard_rate: _X) -> _Y: ...
    def dhf(self, time: _X) -> _Y: ...
    @overload
    def jac_hf(
        self,
        time: _X,
        *,
        asarray: Literal[False],
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _X,
        *,
        asarray: Literal[True],
    ) -> _Y: ...
    def jac_hf(
        self,
        time: _X,
        *,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_chf(
        self,
        time: _X,
        *,
        asarray: Literal[False],
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _X,
        *,
        asarray: Literal[True],
    ) -> _Y: ...
    def jac_chf(
        self,
        time: _X,
        *,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...

class Gompertz(LifetimeDistribution):
    def __init__(self, shape: float | None = None, rate: float | None = None) -> None: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _X) -> _Y: ...
    def chf(self, time: _X) -> _Y: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def mrl(self, time: _X) -> _Y: ...
    @override
    def ichf(self, cumulative_hazard_rate: _X) -> _Y: ...
    def dhf(self, time: _X) -> _Y: ...
    @overload
    def jac_hf(
        self,
        time: _X,
        *,
        asarray: Literal[False],
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _X,
        *,
        asarray: Literal[True],
    ) -> _Y: ...
    def jac_hf(
        self,
        time: _X,
        *,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_chf(
        self,
        time: _X,
        *,
        asarray: Literal[False],
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _X,
        *,
        asarray: Literal[True],
    ) -> _Y: ...
    def jac_chf(
        self,
        time: _X,
        *,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...

class Gamma(LifetimeDistribution):
    def __init__(self, shape: float | None = None, rate: float | None = None) -> None: ...
    def _uppergamma(self, x: _X) -> _Y: ...
    def _jac_uppergamma_shape(self, x: _X) -> _Y: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _X) -> _Y: ...
    def chf(self, time: _X) -> _Y: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def ichf(self, cumulative_hazard_rate: _X) -> _Y: ...
    @override
    def mrl(self, time: _X) -> _Y: ...
    def dhf(self, time: _X) -> _Y: ...
    @overload
    def jac_hf(
        self,
        time: _X,
        *,
        asarray: Literal[False],
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _X,
        *,
        asarray: Literal[True],
    ) -> _Y: ...
    def jac_hf(
        self,
        time: _X,
        *,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_chf(
        self,
        time: _X,
        *,
        asarray: Literal[False],
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _X,
        *,
        asarray: Literal[True],
    ) -> _Y: ...
    def jac_chf(
        self,
        time: _X,
        *,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...

class LogLogistic(LifetimeDistribution):
    def __init__(self, shape: float | None = None, rate: float | None = None) -> None: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: _X) -> _Y: ...
    def chf(self, time: _X) -> _Y: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def ichf(self, cumulative_hazard_rate: _X) -> _Y: ...
    @override
    def mrl(self, time: _X) -> _Y: ...
    def dhf(self, time: _X) -> _Y: ...
    @overload
    def jac_hf(self, time: _X, *, asarray: Literal[False]) -> tuple[_Y, ...]: ...
    @overload
    def jac_hf(self, time: _X, *, asarray: Literal[True]) -> _Y: ...
    def jac_hf(self, time: _X, *, asarray: bool = True) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_chf(self, time: _X, *, asarray: Literal[False]) -> tuple[_Y, ...]: ...
    @overload
    def jac_chf(self, time: _X, *, asarray: Literal[True]) -> _Y: ...
    def jac_chf(self, time: _X, *, asarray: bool = True) -> tuple[_Y, ...] | _Y: ...

class EquilibriumDistribution(ParametricLifetimeModel[*tuple[_X, *_Xs]]):
    baseline: ParametricLifetimeModel[*tuple[_X, *_Xs]]
    fitting_results = Optional[FittingResults]
    def __init__(self, baseline: ParametricLifetimeModel[*tuple[_X, *_Xs]]) -> None: ...
    @property
    def args_names(self) -> tuple[str, ...]: ...
    @override
    def cdf(self, time: _X, *args: *_Xs) -> _Y: ...
    def sf(self, time: _X, *args: *_Xs) -> _Y: ...
    def pdf(self, time: _X, *args: *_Xs) -> _Y: ...
    def hf(self, time: _X, *args: *_Xs) -> _Y: ...
    def chf(self, time: _X, *args: *_Xs) -> _Y: ...
    @override
    def isf(self, probability: _X, *args: *_Xs) -> _Y: ...
    @override
    def ichf(self, cumulative_hazard_rate: _X, *args: *_Xs) -> _Y: ...

class MinimumDistribution(ParametricLifetimeModel[*tuple[int | NDArray[np.int64], *_Xs]]):
    baseline: FittableParametricLifetimeModel[*_Xs]
    fitting_results = Optional[FittingResults]
    def __init__(self, baseline: LifetimeDistribution | LifetimeRegression) -> None: ...
    @override
    def sf(self, time: _X, n: int | NDArray[np.int64], *args: *_Xs) -> _Y: ...
    @override
    def pdf(self, time: _X, n: int | NDArray[np.int64], *args: *_Xs) -> _Y: ...
    @override
    def hf(self, time: _X, n: int | NDArray[np.int64], *args: *_Xs) -> _Y: ...
    @override
    def chf(self, time: _X, n: int | NDArray[np.int64], *args: *_Xs) -> _Y: ...
    @override
    def ichf(
        self,
        cumulative_hazard_rate: _X,
        n: int | NDArray[np.int64],
        *args: *_Xs,
    ) -> _Y: ...
    def dhf(self, time: _X, n: int | NDArray[np.int64], *args: *_Xs) -> _Y: ...
    def jac_chf(
        self,
        time: _X,
        n: int | NDArray[np.int64],
        *args: *_Xs,
        asarray: bool = False,
    ) -> _Y | tuple[_Y, ...]: ...
    def jac_hf(
        self,
        time: _X,
        n: int | NDArray[np.int64],
        *args: *_Xs,
        asarray: bool = False,
    ) -> _Y | tuple[_Y, ...]: ...
    def jac_sf(
        self,
        time: _X,
        n: int | NDArray[np.int64],
        *args: *_Xs,
        asarray: bool = False,
    ) -> _Y | tuple[_Y, ...]: ...
    def jac_cdf(
        self,
        time: _X,
        n: int | NDArray[np.int64],
        *args: *_Xs,
        asarray: bool = False,
    ) -> _Y | tuple[_Y, ...]: ...
    def jac_pdf(
        self,
        time: _X,
        n: int | NDArray[np.int64],
        *args: *_Xs,
        asarray: bool = False,
    ) -> _Y | tuple[_Y, ...]: ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_X], _Y],
        a: _X,
        b: _X,
        n: int | NDArray[np.int64],
        *args: *_Xs,
        deg: int = 10,
    ) -> _Y: ...
    def fit(
        self,
        time: NDArray[np.float64],
        n: NDArray[np.int64],
        *args: NDArray[np.float64],
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
        departure: NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> Self: ...
