"""Additional types used in the codebase for type checking."""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Protocol,
    TypeAlias,
    TypeVarTuple,
    overload,
)

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from relife.lifetime_model._distribution import LifetimeDistribution
    from relife.lifetime_model._frozen import FrozenParametricLifetimeModel

from ._random import Seed
from ._scalars import AnyFloat, NumpyBool, NumpyFloat

__all__ = ["AnyParametricLifetimeModel", "AnyLifetimeDistribution"]

Ts = TypeVarTuple("Ts")


class AnyParametricLifetimeModel(Protocol[*Ts]):
    """
    Structural type for any parametric lifetime model.
    It is particularly needed where an argument can expect both
    ParametricLifetimeModel[*Ts] and FrozenParametricLifetimeModel[*Ts]
    whereas both interface are different (the first expects args but the second
    doesn't). See conditional_model.py.
    """

    def sf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat: ...
    def hf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat: ...
    def chf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat: ...
    def pdf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat: ...
    def cdf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat: ...
    def ppf(self, probability: AnyFloat, *args: *Ts) -> NumpyFloat: ...
    def median(self, *args: *Ts) -> NumpyFloat: ...
    def isf(self, probability: AnyFloat, *args: *Ts) -> NumpyFloat: ...
    def ichf(self, cumulative_hazard_rate: AnyFloat, *args: *Ts) -> NumpyFloat: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> NumpyFloat: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Seed | None = None,
    ) -> (
        NumpyFloat
        | tuple[NumpyFloat, NumpyBool]
        | tuple[NumpyFloat, NumpyFloat]
        | tuple[NumpyFloat, NumpyBool, NumpyFloat]
    ): ...
    def rvs(
        self,
        size: int,
        *args: *Ts,
        nb_assets: int | None = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Seed | None = None,
    ) -> (
        NumpyFloat
        | tuple[NumpyFloat, NumpyBool]
        | tuple[NumpyFloat, NumpyFloat]
        | tuple[NumpyFloat, NumpyBool, NumpyFloat]
    ): ...
    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: NumpyFloat,
        b: NumpyFloat,
        *args: *Ts,
        deg: int = 10,
    ) -> NumpyFloat: ...
    def moment(self, n: int, *args: *Ts) -> NumpyFloat: ...
    def mean(self, *args: *Ts) -> NumpyFloat: ...
    def var(self, *args: *Ts) -> NumpyFloat: ...
    def mrl(self, time: AnyFloat, *args: *Ts) -> NumpyFloat: ...


AnyLifetimeDistribution: TypeAlias = FrozenParametricLifetimeModel[*tuple[Any, ...]] | LifetimeDistribution
