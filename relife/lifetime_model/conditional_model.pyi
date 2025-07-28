import numpy as np
from ._base import (
    FrozenParametricLifetimeModel as FrozenParametricLifetimeModel,
    ParametricLifetimeModel as ParametricLifetimeModel,
)
from _typeshed import Incomplete
from numpy.typing import NDArray
from typing import Callable, Literal, overload
from typing_extensions import override

def reshape_ar_or_a0(name: str, value: float | NDArray[np.float64]) -> NDArray[np.float64]: ...

class AgeReplacementModel(
    ParametricLifetimeModel[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]
):
    baseline: Incomplete
    def __init__(
        self,
        baseline: (
            ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]
            | FrozenParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]
        ),
    ) -> None: ...
    def sf(
        self, time: float | NDArray[np.float64], ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> NDArray[np.float64]: ...
    def hf(
        self, time: float | NDArray[np.float64], ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> NDArray[np.float64]: ...
    def chf(
        self, time: float | NDArray[np.float64], ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> NDArray[np.float64]: ...
    @override
    def isf(
        self,
        probability: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]: ...
    @override
    def ichf(
        self,
        cumulative_hazard_rate: NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]: ...
    def pdf(
        self, time: float | NDArray[np.float64], ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> NDArray[np.float64]: ...
    def ppf(
        self,
        probability: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> NDArray[np.float64]: ...
    def median(self, ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]) -> NDArray[np.float64]: ...
    @override
    def mrl(
        self, time: float | NDArray[np.float64], ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> NDArray[np.float64]: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: float | NDArray[np.float64],
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
        ar: float | NDArray[np.float64],
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
        ar: float | NDArray[np.float64],
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
        ar: float | NDArray[np.float64],
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
        ar: float | NDArray[np.float64],
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
    @override
    def ls_integrate(
        self,
        func: Callable[[float | np.float64 | NDArray[np.float64]], float | np.float64 | NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        ar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        deg: int = 10,
    ) -> NDArray[np.float64]: ...
    @override
    def moment(
        self, n: int, ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> NDArray[np.float64]: ...
    @override
    def mean(self, ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]) -> NDArray[np.float64]: ...
    @override
    def var(self, ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]) -> NDArray[np.float64]: ...

class LeftTruncatedModel(
    ParametricLifetimeModel[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]
):
    baseline: Incomplete
    def __init__(
        self,
        baseline: (
            ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]
            | FrozenParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]
        ),
    ) -> None: ...
    def sf(
        self, time: float | NDArray[np.float64], a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    def pdf(
        self, time: float | NDArray[np.float64], a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    def isf(
        self, probability: NDArray[np.float64], a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    def chf(
        self, time: float | NDArray[np.float64], a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    def cdf(
        self, time: float | NDArray[np.float64], ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    def hf(
        self, time: float | NDArray[np.float64], a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]: ...
    @overload
    def rvs(
        self,
        size: int,
        a0: float | NDArray[np.float64],
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
        a0: float | NDArray[np.float64],
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
        a0: float | NDArray[np.float64],
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
        a0: float | NDArray[np.float64],
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
        a0: float | NDArray[np.float64],
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
    @override
    def ls_integrate(
        self,
        func: Callable[[float | np.float64 | NDArray[np.float64]], float | np.float64 | NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        deg: int = 10,
    ) -> NDArray[np.float64]: ...
    @override
    def mean(
        self, a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def median(
        self, a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def var(
        self, a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def moment(
        self, n: int, a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def mrl(
        self, time: float | NDArray[np.float64], a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]: ...
    @override
    def ppf(
        self,
        probability: float | NDArray[np.float64],
        a0: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]: ...

class FrozenAgeReplacementModel(
    FrozenParametricLifetimeModel[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]
):
    unfrozen_model: AgeReplacementModel
    frozen_args: tuple[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]
    @override
    def __init__(
        self, model: AgeReplacementModel, ar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> None: ...
    @override
    def unfreeze(self) -> AgeReplacementModel: ...
    @property
    def ar(self) -> float | NDArray[np.float64]: ...
    args: Incomplete
    @ar.setter
    def ar(self, value: float | NDArray[np.float64]) -> None: ...

class FrozenLeftTruncatedModel(
    FrozenParametricLifetimeModel[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]
):
    unfrozen_model: LeftTruncatedModel
    frozen_args: tuple[float | NDArray[np.float64], *tuple[float | NDArray[np.float64], ...]]
    @override
    def __init__(
        self, model: LeftTruncatedModel, a0: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> None: ...
    @override
    def unfreeze(self) -> LeftTruncatedModel: ...
    @property
    def a0(self) -> float | NDArray[np.float64]: ...
    args: Incomplete
    @a0.setter
    def a0(self, value: float | NDArray[np.float64]) -> None: ...

A0_TIME_BASE_DOCSTRING: str
A0_MOMENT_BASE_DOCSTRING: str
A0_PROBABILITY_BASE_DOCSTRING: str
AR_TIME_BASE_DOCSTRING: str
AR_MOMENT_BASE_DOCSTRING: str
AR_PROBABILITY_BASE_DOCSTRING: str
