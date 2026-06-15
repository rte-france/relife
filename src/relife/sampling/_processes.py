"""Stochastic process sampling functions."""

from __future__ import annotations

from functools import singledispatch
from typing import (
    TypeAlias,
    TypeVarTuple,
    overload,
)

import numpy as np
from optype.numpy import Array1D

from relife.stochastic_processes import (
    FrozenKijima1Process,
    FrozenKijima2Process,
    FrozenNonHomogeneousPoissonProcess,
    RenewalProcess,
)

from ._data import StochasticSampleMapping
from ._iterables import (
    Kijima1ProcessIterable,
    Kijima2ProcessIterable,
    NonHomogeneousPoissonProcessIterable,
    RenewalProcessIterable,
)

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint
AgeArg: TypeAlias = ST | NumpyST | Array1D[NumpyST] | None
Seed: TypeAlias = (
    int | np.random.Generator | np.random.BitGenerator | np.random.RandomState | None
)
Ts = TypeVarTuple("Ts")

__all__ = ["sample_process"]


@overload
def sample_process(
    process: RenewalProcess,
    nb_samples: int,
    time_window: tuple[float, float],
    *,
    a0: AgeArg = None,
    ar: AgeArg = None,
    seed: Seed = None,
) -> StochasticSampleMapping: ...
@overload
def sample_process(
    process: FrozenNonHomogeneousPoissonProcess[*Ts],
    nb_samples: int,
    time_window: tuple[float, float],
    a0: AgeArg = None,
    ar: AgeArg = None,
    seed: Seed = None,
) -> StochasticSampleMapping: ...
@overload
def sample_process(
    process: FrozenKijima1Process[*Ts],
    nb_samples: int,
    time_window: tuple[float, float],
    a0: AgeArg = None,
    ar: AgeArg = None,
    seed: Seed = None,
) -> StochasticSampleMapping: ...
@overload
def sample_process(
    process: FrozenKijima2Process[*Ts],
    nb_samples: int,
    time_window: tuple[float, float],
    a0: AgeArg = None,
    ar: AgeArg = None,
    seed: Seed = None,
) -> StochasticSampleMapping: ...
def sample_process(
    process: RenewalProcess
    | FrozenNonHomogeneousPoissonProcess[*Ts]
    | FrozenKijima1Process[*Ts]
    | FrozenKijima2Process[*Ts],
    nb_samples: int,
    time_window: tuple[float, float],
    a0: AgeArg = None,
    ar: AgeArg = None,
    seed: Seed = None,
) -> StochasticSampleMapping:
    """Sample paths from a fully specified stochastic process."""
    return _sample_process(process, nb_samples, time_window, a0=a0, ar=ar, seed=seed)


def _sample_from_iterable(
    iterable: RenewalProcessIterable
    | NonHomogeneousPoissonProcessIterable
    | Kijima1ProcessIterable
    | Kijima2ProcessIterable,
    nb_samples: int,
) -> StochasticSampleMapping:
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))
    return StochasticSampleMapping.from_struct_array(
        struct_array, iterable.nb_assets, nb_samples
    )


@singledispatch
def _sample_process(
    process: RenewalProcess,
    nb_samples: int,
    time_window: tuple[float, float],
    a0: AgeArg = None,
    ar: AgeArg = None,
    seed: Seed = None,
) -> StochasticSampleMapping:
    raise ValueError("Invalid process argument.")


@_sample_process.register
def _(
    process: RenewalProcess,
    nb_samples: int,
    time_window: tuple[float, float],
    a0: AgeArg = None,
    ar: AgeArg = None,
    seed: Seed = None,
) -> StochasticSampleMapping:
    """Sample paths from a renewal or renewal reward process."""
    iterable = RenewalProcessIterable(
        process, nb_samples, time_window, a0=a0, ar=ar, seed=seed
    )
    return _sample_from_iterable(iterable, nb_samples)


@_sample_process.register(FrozenNonHomogeneousPoissonProcess)
def _(
    process: FrozenNonHomogeneousPoissonProcess[*Ts],
    nb_samples: int,
    time_window: tuple[float, float],
    a0: AgeArg = None,
    ar: AgeArg = None,
    seed: Seed = None,
) -> StochasticSampleMapping:
    """Sample paths from a frozen non-homogeneous Poisson process."""
    iterable = NonHomogeneousPoissonProcessIterable(
        process,
        nb_samples,
        time_window=time_window,
        a0=a0,
        ar=ar,
        seed=seed,
    )
    return _sample_from_iterable(iterable, nb_samples)


@_sample_process.register(FrozenKijima1Process)
def _(
    process: FrozenKijima1Process[*Ts],
    nb_samples: int,
    time_window: tuple[float, float],
    a0: AgeArg = None,
    ar: AgeArg = None,
    seed: Seed = None,
) -> StochasticSampleMapping:
    """Sample paths from a frozen Kijima type I process."""
    iterable = Kijima1ProcessIterable(
        process,
        nb_samples,
        time_window=time_window,
        a0=a0,
        ar=ar,
        seed=seed,
    )
    return _sample_from_iterable(iterable, nb_samples)


@_sample_process.register(FrozenKijima2Process)
def _(
    process: FrozenKijima2Process[*Ts],
    nb_samples: int,
    time_window: tuple[float, float],
    a0: AgeArg = None,
    ar: AgeArg = None,
    seed: Seed = None,
) -> StochasticSampleMapping:
    """Sample paths from a frozen Kijima type II process."""
    iterable = Kijima2ProcessIterable(
        process,
        nb_samples,
        time_window=time_window,
        a0=a0,
        ar=ar,
        seed=seed,
    )
    return _sample_from_iterable(iterable, nb_samples)
