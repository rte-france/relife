from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife.utils import get_model_nb_assets

from ._iterators import (
    _NonHomogeneousPoissonProcessIterator,
    _RenewalProcessIterator,
    _RenewalRewardProcessIterator,
    _StochasticDataIterator,
)


class StochasticDataIterable(Iterable[NDArray[np.void]], ABC):
    def __init__(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        seed: Optional[int] = None,
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

    @abstractmethod
    @override
    def __iter__(self) -> _StochasticDataIterator: ...


class RenewalProcessIterable(StochasticDataIterable):
    def __init__(
        self,
        process,
        nb_samples: int,
        time_window: tuple[float, float],
        seed: Optional[int] = None,
    ):
        super().__init__(nb_samples, time_window, seed=seed)
        self.process = process

    def __iter__(self) -> _RenewalProcessIterator:
        from relife.stochastic_process import RenewalProcess, RenewalRewardProcess

        if isinstance(self.process, RenewalProcess):
            return _RenewalProcessIterator(
                self.process,
                self.nb_samples,
                self.time_window,
                nb_assets=get_model_nb_assets(self.process),
                seed=self.seed,
            )
        if isinstance(self.process, RenewalRewardProcess):
            return _RenewalRewardProcessIterator(
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
        seed: Optional[int] = None,
    ):
        super().__init__(nb_samples, time_window, seed=seed)
        self.process = process

    def __iter__(self) -> _NonHomogeneousPoissonProcessIterator:
        return _NonHomogeneousPoissonProcessIterator(
            self.process,
            self.nb_samples,
            self.time_window,
            nb_assets=get_model_nb_assets(self.process),
            seed=self.seed,
        )
