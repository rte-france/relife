from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, Self, TypeAlias, TypeVarTuple

import numpy as np
from optype.numpy import Array1D, ArrayND

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint

Ts = TypeVarTuple("Ts")


class AnyParametricLifetimeModel(Protocol[*Ts]):
    def sf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]: ...

    def hf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]: ...

    def chf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]: ...

    def pdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]: ...

    def cdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]: ...

    def ppf(
        self, probability: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]: ...

    def median(self, *args: *Ts) -> np.float64 | ArrayND[np.float64]: ...

    def isf(
        self, probability: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]: ...

    def ichf(
        self,
        cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]: ...

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *args: *Ts,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> np.float64 | ArrayND[np.float64]: ...

    def ls_integrate(
        self,
        func: Callable[
            [ST | NumpyST | ArrayND[NumpyST]],
            np.float64 | ArrayND[np.float64],
        ],
        a: ST | NumpyST | ArrayND[NumpyST],
        b: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
        deg: int = 10,
    ) -> np.float64 | ArrayND[np.float64]: ...

    def moment(
        self,
        n: int,
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]: ...

    def mean(self, *args: *Ts) -> np.float64 | ArrayND[np.float64]: ...

    def var(self, *args: *Ts) -> np.float64 | ArrayND[np.float64]: ...

    def mrl(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]: ...

    def apply_condition(
        self,
        *,
        ar: ST | NumpyST | ArrayND[NumpyST] | None = None,
        a0: ST | NumpyST | ArrayND[NumpyST] | None = None,
    ) -> AnyParametricLifetimeModel[*Ts]: ...

    def freeze(self, *args: *Ts) -> AnyParametricLifetimeModel[()]: ...


class FittableParametricLifetimeModel(Protocol[*Ts]):
    """
    A structural type for any ParametricLifetimeModel having a fit method.
    """

    def fit(
        self,
        time: Array1D[np.float64],
        *args: Any,
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> Self: ...


class DifferentiableParametricLifetimeModel(Protocol[*Ts]):
    def dhf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> ArrayND[np.float64]: ...

    def jac_chf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> ArrayND[np.float64]: ...

    def jac_hf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> ArrayND[np.float64]: ...

    def jac_sf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> ArrayND[np.float64]: ...

    def jac_cdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> ArrayND[np.float64]: ...

    def jac_pdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> ArrayND[np.float64]: ...
