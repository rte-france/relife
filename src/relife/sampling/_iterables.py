from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from optype.numpy import Array1D
from typing_extensions import override

from relife.typing import ST, NumpyST

from ._iterators import (
    Kijima1ProcessIterator,
    Kijima2ProcessIterator,
    NonHomogeneousPoissonProcessIterator,
    RenewalProcessIterator,
    RenewalRewardProcessIterator,
    StochasticDataIterator,
)


class StochasticDataIterable(Iterable[NDArray[np.void]], ABC):
    nb_samples: int
    a0: ST | NumpyST | Array1D[NumpyST] | None
    ar: ST | NumpyST | Array1D[NumpyST] | None
    seed: (
        int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None
    )

    def __init__(
        self,
        process,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ):
        self.process = process

        t0, tf = time_window
        if t0 < 0 or tf < 0 or t0 > tf:
            raise ValueError(
                f"Incorrect time window. Got {time_window}. Values must be positive and first value can't lower than second value."  # noqa: E501
            )
        self.time_window = t0, tf

        self.a0 = a0
        self.ar = ar
        self.nb_samples = nb_samples
        self.seed = seed

    @property
    def t0(self) -> float:
        return self.time_window[0]

    @property
    def tf(self) -> float:
        return self.time_window[1]

    @override
    @abstractmethod
    def __iter__(self) -> StochasticDataIterator: ...


class RenewalProcessIterable(StochasticDataIterable):
    @override
    def __iter__(self) -> RenewalProcessIterator:
        from relife.stochastic_processes import RenewalProcess, RenewalRewardProcess

        if isinstance(self.process, RenewalRewardProcess):
            return RenewalRewardProcessIterator(
                self.process,
                self.nb_samples,
                self.time_window,
                self.a0,
                self.ar,
                seed=self.seed,
            )
        if isinstance(self.process, RenewalProcess):
            return RenewalProcessIterator(
                self.process,
                self.nb_samples,
                self.time_window,
                self.a0,
                self.ar,
                seed=self.seed,
            )
        raise ValueError


class NonHomogeneousPoissonProcessIterable(StochasticDataIterable):
    @override
    def __iter__(self) -> NonHomogeneousPoissonProcessIterator:
        return NonHomogeneousPoissonProcessIterator(
            self.process,
            self.nb_samples,
            self.time_window,
            a0=self.a0,
            ar=self.ar,
            seed=self.seed,
        )


class Kijima1ProcessIterable(StochasticDataIterable):
    @override
    def __iter__(self) -> Kijima1ProcessIterator:
        return Kijima1ProcessIterator(
            self.process,
            self.nb_samples,
            self.time_window,
            a0=self.a0,
            ar=self.ar,
            seed=self.seed,
        )


class Kijima2ProcessIterable(StochasticDataIterable):
    @override
    def __iter__(self) -> Kijima2ProcessIterator:
        return Kijima2ProcessIterator(
            self.process,
            self.nb_samples,
            self.time_window,
            a0=self.a0,
            ar=self.ar,
            seed=self.seed,
        )
