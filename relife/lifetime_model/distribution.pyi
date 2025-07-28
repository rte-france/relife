import abc
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import Bounds

from relife.lifetime_model import LifetimeRegression
from ._base import ParametricLifetimeModel, FittableParametricLifetimeModel
from _typeshed import Incomplete
from numpy.typing import NDArray
from typing import Any, Callable, Literal, Self, overload, Optional
from typing_extensions import override

from ..data import LifetimeData
from relife.likelihood import FittingResults


class LifetimeDistribution(FittableParametricLifetimeModel[()], ABC):
    fitting_results: Optional[FittingResults]

    @override
    def sf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @override
    def isf(self, probability: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @override
    def cdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def pdf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def ppf(self, probability: float) -> np.float64: ...
    @overload
    def ppf(self, probability: NDArray[np.float64]) -> NDArray[np.float64]: ...
    @overload
    def ppf(self, probability: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @override
    def median(self) -> np.float64: ...
    @abstractmethod
    def dhf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False]
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True]
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False]
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True]
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_sf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False]
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_sf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True]
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_cdf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False] = False
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_cdf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True] = True
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_pdf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False] = False
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_pdf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True] = True
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def moment(self, n: int) -> np.float64: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: int | None = None,
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[False],
        seed: int | None = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_]]: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[True],
        seed: int | None = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[True],
        seed: int | None = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_], np.float64 | NDArray[np.float64]]: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
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
    @override
    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], np.float64 | NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        deg: int = 10,
    ) -> np.float64 | NDArray[np.float64]: ...


    def _init_params(self, lifetime_data : LifetimeData) -> None: ...
    def _params_bounds(self) -> Bounds:

    @override
    def fit(
        self,
        time: NDArray[np.float64],
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
        departure: NDArray[np.float64] | None = None,
        **options: Any,
    ) -> Self: ...


class Exponential(LifetimeDistribution):
    def __init__(self, rate: float | None = None) -> None: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def chf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @override
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False]
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True]
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False]
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True]
    ) -> np.float64 | NDArray[np.float64]: ...
    def dhf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...

class Weibull(LifetimeDistribution):
    def __init__(self, shape: float | None = None, rate: float | None = None) -> None: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def chf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @override
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False]
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True]
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False]
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True]
    ) -> np.float64 | NDArray[np.float64]: ...
    def dhf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...

class Gompertz(LifetimeDistribution):
    def __init__(self, shape: float | None = None, rate: float | None = None) -> None: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def chf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @override
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False]
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True]
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False]
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True]
    ) -> np.float64 | NDArray[np.float64]: ...
    def dhf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...

class Gamma(LifetimeDistribution):
    def __init__(self, shape: float | None = None, rate: float | None = None) -> None: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def chf(self, time: float | NDArray[np.float64]) -> float | NDArray[np.float64]: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False]
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True]
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False]
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True]
    ) -> np.float64 | NDArray[np.float64]: ...
    def dhf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @override
    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...

class LogLogistic(LifetimeDistribution):
    def __init__(self, shape: float | None = None, rate: float | None = None) -> None: ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...
    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def chf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @override
    def mean(self) -> np.float64: ...
    @override
    def var(self) -> np.float64: ...
    @override
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False]
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True]
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[False]
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: Literal[True]
    ) -> np.float64 | NDArray[np.float64]: ...
    def dhf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @override
    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...

class EquilibriumDistribution(ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]):
    baseline: Incomplete
    def __init__(self, baseline: ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]) -> None: ...
    @property
    def args_names(self) -> tuple[str, ...]: ...
    @override
    def cdf(
        self, time: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    def sf(
        self, time: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    def pdf(
        self, time: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    def hf(
        self, time: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    def chf(
        self, time: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def isf(
        self, probability: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...

class MinimumDistribution(ParametricLifetimeModel[int | NDArray[np.int64], *tuple[float | NDArray[np.float64], ...]]):
    baseline: Incomplete
    def __init__(self, baseline: LifetimeDistribution | LifetimeRegression) -> None: ...
    @override
    def sf(
        self, time: float | NDArray[np.float64], n: int | NDArray[np.int64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def pdf(
        self, time: float | NDArray[np.float64], n: int | NDArray[np.int64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def hf(
        self, time: float | NDArray[np.float64], n: int | NDArray[np.int64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def chf(
        self, time: float | NDArray[np.float64], n: int | NDArray[np.int64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        n: int | NDArray[np.int64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]: ...
    def dhf(
        self, time: float | NDArray[np.float64], n: int | NDArray[np.int64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        n: int | NDArray[np.int64],
        *args: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]: ...
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        n: int | NDArray[np.int64],
        *args: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]: ...
    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        n: int | NDArray[np.int64],
        *args: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]: ...
    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        n: int | NDArray[np.int64],
        *args: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]: ...
    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        n: int | NDArray[np.int64],
        *args: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]: ...
    @override
    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], np.float64 | NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        n: int | NDArray[np.int64],
        *args: float | NDArray[np.float64],
        deg: int = 10,
    ) -> np.float64 | NDArray[np.float64]: ...
    params: Incomplete
    fitting_results: Incomplete
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

