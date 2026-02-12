# pyright: basic

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife.utils import get_model_nb_assets

from ._iterators import (
    Kijima1ProcessIterator,
    Kijima2ProcessIterator,
    NonHomogeneousPoissonProcessIterator,
    RenewalProcessIterator,
    RenewalRewardProcessIterator,
    StochasticDataIterator,
)

__all__ = [
    "StochasticDataIterable",
    "RenewalProcessIterable",
    "NonHomogeneousPoissonProcessIterable",
    "Kijima1ProcessIterable",
    "Kijima2ProcessIterable",
]


class StochasticDataIterable(Iterable[NDArray[np.void]], ABC):
    def __init__(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        seed: int | None = None,
    ):
        t0, tf = time_window
        if t0 < 0 or tf < 0 or t0 > tf:
            raise ValueError(
                f"Incorrect time window. Got {time_window}. Values must be positive and first value can't lower than second value."
            )
        self.time_window = t0, tf
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
    def __init__(
        self,
        process,
        nb_samples: int,
        time_window: tuple[float, float],
        seed: int | None = None,
    ):
        super().__init__(nb_samples, time_window, seed=seed)
        self.process = process

    def __iter__(self) -> RenewalProcessIterator:
        from relife.stochastic_process import RenewalProcess, RenewalRewardProcess

        if isinstance(self.process, RenewalProcess):
            return RenewalProcessIterator(
                self.process,
                self.nb_samples,
                self.time_window,
                nb_assets=get_model_nb_assets(self.process),
                seed=self.seed,
            )
        if isinstance(self.process, RenewalRewardProcess):
            return RenewalRewardProcessIterator(
                self.process,
                self.nb_samples,
                self.time_window,
                nb_assets=get_model_nb_assets(self.process),
                seed=self.seed,
            )
        raise ValueError


class NonHomogeneousPoissonProcessIterable(StochasticDataIterable):
    def __init__(
        self,
        process,
        nb_samples: int,
        time_window: tuple[float, float],
        seed: int | None = None,
    ):
        super().__init__(nb_samples, time_window, seed=seed)
        self.process = process

    def __iter__(self) -> NonHomogeneousPoissonProcessIterator:
        return NonHomogeneousPoissonProcessIterator(
            self.process,
            self.nb_samples,
            self.time_window,
            nb_assets=get_model_nb_assets(self.process),
            seed=self.seed,
        )


class Kijima1ProcessIterable(StochasticDataIterable):
    def __init__(
        self,
        process,
        nb_samples: int,
        time_window: tuple[float, float],
        seed: int | None = None,
    ):
        super().__init__(nb_samples, time_window, seed=seed)
        self.process = process

    def __iter__(self) -> Kijima1ProcessIterator:
        return Kijima1ProcessIterator(
            self.process,
            self.nb_samples,
            self.time_window,
            nb_assets=get_model_nb_assets(self.process),
            seed=self.seed,
        )


class Kijima2ProcessIterable(StochasticDataIterable):
    def __init__(
        self,
        process,
        nb_samples: int,
        time_window: tuple[float, float],
        seed: int | None = None,
    ):
        super().__init__(nb_samples, time_window, seed=seed)
        self.process = process

    def __iter__(self) -> Kijima2ProcessIterator:
        return Kijima2ProcessIterator(
            self.process,
            self.nb_samples,
            self.time_window,
            nb_assets=get_model_nb_assets(self.process),
            seed=self.seed,
        )
