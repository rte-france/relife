from _typeshed import Incomplete
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from relife import ParametricModel as ParametricModel
from relife.lifetime_model import FittableParametricLifetimeModel, FrozenParametricLifetimeModel, LifetimeDistribution
from typing import Literal, overload, Callable, Self
from typing_extensions import override

from relife.likelihood import FittingResults as FittingResults


def broadcast_time_covar(
    time: float | NDArray[np.float64], covar: float | NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def broadcast_time_covar_shapes(
    time_shape: tuple[()] | tuple[int] | tuple[int, int], covar_shape: tuple[()] | tuple[int] | tuple[int, int]
) -> tuple[()] | tuple[int] | tuple[int, int]: ...


class LifetimeRegression(
    FittableParametricLifetimeModel[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]],
    ABC,
):
    fitting_results = FittingResults | None
    covar_effect: Incomplete
    baseline: Incomplete
    def __init__(
        self, baseline: LifetimeDistribution | LifetimeRegression, coefficients: tuple[float | None, ...] = (None,)
    ) -> None: ...
    @property
    def coefficients(self) -> NDArray[np.float64]: ...
    @property
    def nb_coef(self) -> int: ...
    @override
    def sf(
        self, time: float | NDArray[np.float64], covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def isf(
        self,
        probability: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def cdf(
        self, time: float | NDArray[np.float64], covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    def pdf(
        self, time: float | NDArray[np.float64], covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def ppf(
        self,
        probability: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def mrl(
        self, time: float | NDArray[np.float64], covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], np.float64 | NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        deg: int = 10,
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def mean(
        self, covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def var(
        self, covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def median(
        self, covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @abstractmethod
    def dhf(
        self, time: float | NDArray[np.float64], covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False],
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True],
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: bool,
    ) -> tuple[np.float64 | NDArray[np.float64], ...] | np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def rvs(
        self,
        size: int,
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: int | None = None,
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def rvs(
        self,
        size: int,
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[False],
        seed: int | None = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_]]: ...
    @overload
    def rvs(
        self,
        size: int,
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[True],
        seed: int | None = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]: ...
    @overload
    def rvs(
        self,
        size: int,
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[True],
        seed: int | None = None,
    ) -> tuple[np.float64 | NDArray[np.float64], np.bool_ | NDArray[np.bool_], np.float64 | NDArray[np.float64]]: ...
    @overload
    def rvs(
        self,
        size: int,
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
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
    params: Incomplete
    def fit(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
        departure: NDArray[np.float64] | None = None,
        options: dict = {},
        **kwargs: NDArray[np.float64],
    ) -> Self: ...
    @override
    def freeze(self, covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]): ...


class ProportionalHazard(LifetimeRegression):
    def hf(
        self, time: float | NDArray[np.float64], covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    def chf(
        self, time: float | NDArray[np.float64], covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]: ...
    def dhf(
        self, time: float | NDArray[np.float64], covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def moment(
        self, n: int, covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...

class AcceleratedFailureTime(LifetimeRegression):
    def hf(
        self, time: float | NDArray[np.float64], covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    def chf(
        self, time: float | NDArray[np.float64], covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]: ...
    def dhf(
        self, time: float | NDArray[np.float64], covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def moment(
        self, n: int, covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...


class CovarEffect(ParametricModel):
    def __init__(self, coefficients: tuple[float | None, ...] = (None,)) -> None: ...
    @property
    def nb_coef(self) -> int: ...
    def g(self, covar: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    def jac_g(
        self, covar: float | NDArray[np.float64], *, asarray: bool = False
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]: ...


class FrozenLifetimeRegression(
    FrozenParametricLifetimeModel[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]
):
    unfrozen_model: LifetimeRegression

    @override
    def __init__(
        self, model: LifetimeRegression, covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> None: ...
    @override
    def unfreeze(self) -> LifetimeRegression: ...
    @property
    def nb_coef(self) -> int: ...
    @property
    def covar(self) -> float | NDArray[np.float64]: ...
    args: Incomplete
    @covar.setter
    def covar(self, value: float | NDArray[np.float64]) -> None: ...
    def dhf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_hf(
        self, time: float | NDArray[np.float64], asarray: Literal[False] = False
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_hf(
        self, time: float | NDArray[np.float64], asarray: Literal[True] = True
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_chf(
        self, time: float | NDArray[np.float64], asarray: Literal[False] = False
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_chf(
        self, time: float | NDArray[np.float64], asarray: Literal[True] = True
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_sf(
        self, time: float | NDArray[np.float64], asarray: Literal[False] = False
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_sf(
        self, time: float | NDArray[np.float64], asarray: Literal[True] = True
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_cdf(
        self, time: float | NDArray[np.float64], asarray: Literal[False]
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_cdf(
        self, time: float | NDArray[np.float64], asarray: Literal[True]
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_cdf(
        self, time: float | NDArray[np.float64], asarray: bool
    ) -> tuple[np.float64 | NDArray[np.float64], ...] | np.float64 | NDArray[np.float64]: ...
    @overload
    def jac_pdf(
        self, time: float | NDArray[np.float64], asarray: Literal[False] = False
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...
    @overload
    def jac_pdf(
        self, time: float | NDArray[np.float64], asarray: Literal[True] = True
    ) -> np.float64 | NDArray[np.float64]: ...
