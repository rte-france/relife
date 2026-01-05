"""Additional types used in the codebase for type checking."""

from typing import (
    Callable,
    Literal,
    Protocol,
    TypeVarTuple,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from ._random import Seed
from ._scalars import AnyFloat, NumpyBool, NumpyFloat

__all__ = ["AnyParametricLifetimeModel"]

Ts = TypeVarTuple("Ts")


class AnyParametricLifetimeModel(Protocol[*Ts]):
    """
    Structural type for any parametric lifetime model.
    It is particularly needed where an parameter can expect
    ParametricLifetimeModel[*Ts] or FrozenParametricLifetimeModel[*Ts]
    Their interfaces are different (the first expects args but the second
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
