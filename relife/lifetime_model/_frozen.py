from typing import Callable, Literal, TypeVarTuple, overload

import numpy as np
from numpy.typing import NDArray

from relife.base import FrozenParametricModel
from relife.typing import AnyFloat, NumpyBool, NumpyFloat, Seed

from ._base import ParametricLifetimeModel

Ts = TypeVarTuple("Ts")

__all__ = ["FrozenParametricLifetimeModel"]

Ts = TypeVarTuple("Ts")


class FrozenParametricLifetimeModel(FrozenParametricModel[ParametricLifetimeModel[*Ts], *Ts]):

    _args: tuple[*Ts]
    _unfrozen_model: ParametricLifetimeModel[*Ts]

    def __init__(self, model: ParametricLifetimeModel[*Ts], *args: *Ts) -> None:
        super().__init__(model, *args)

    def sf(self, time: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.sf(time, *self._args)

    def hf(self, time: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.hf(time, *self._args)

    def chf(self, time: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.chf(time, *self._args)

    def pdf(self, time: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.pdf(time, *self._args)

    def cdf(self, time: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.cdf(time, *self._args)

    def ppf(self, probability: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.ppf(probability, *self._args)

    def median(self) -> NumpyFloat:
        return self._unfrozen_model.median(*self._args)

    def isf(self, probability: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.isf(probability, *self._args)

    def ichf(self, cumulative_hazard_rate: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.ichf(cumulative_hazard_rate, *self._args)

    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> NumpyFloat: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool]: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
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
        *,
        nb_assets: int | None = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Seed | None = None,
    ) -> (
        NumpyFloat
        | tuple[NumpyFloat, NumpyBool]
        | tuple[NumpyFloat, NumpyFloat]
        | tuple[NumpyFloat, NumpyBool, NumpyFloat]
    ):
        return self._unfrozen_model.rvs(
            size, *self._args, nb_assets=nb_assets, return_event=return_event, return_entry=return_entry, seed=seed
        )

    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: NumpyFloat,
        b: NumpyFloat,
        *,
        deg: int = 10,
    ) -> NumpyFloat:
        return self._unfrozen_model.ls_integrate(func, a, b, *self._args, deg=deg)

    def moment(self, n: int) -> NumpyFloat:
        return self._unfrozen_model.moment(n, *self._args)

    def mean(self) -> NumpyFloat:
        return self._unfrozen_model.mean(*self._args)

    def var(self) -> NumpyFloat:
        return self._unfrozen_model.var(*self._args)

    def mrl(self, time: AnyFloat) -> NumpyFloat:
        return self._unfrozen_model.mrl(time, *self._args)
