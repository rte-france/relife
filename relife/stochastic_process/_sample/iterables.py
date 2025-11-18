from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Iterable,
    Optional,
    TypeVarTuple,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from .iterators import (
    StochasticDataIterator,
    NonHomogeneousPoissonProcessIterator,
    RenewalProcessIterator,
    RenewalRewardProcessIterator,
)

if TYPE_CHECKING:
    from relife.stochastic_process import RenewalProcess

Args = TypeVarTuple("Args")


class StochasticDataIterable(Iterable[NDArray[np.void]], ABC):
    def __init__(
        self,
        nb_samples: int,
        tf: float,
        t0: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.tf = tf
        self.t0 = t0
        self.nb_samples = nb_samples
        self.seed = seed

    @abstractmethod
    @override
    def __iter__(self) -> StochasticDataIterator: ...


class RenewalProcessIterable(StochasticDataIterable):
    def __init__(
        self,
        process,
        nb_samples: int,
        tf: float,
        t0: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(nb_samples, tf, t0=t0, seed=seed)
        self.process = process

    def __iter__(self) -> RenewalProcessIterator:
        from relife.stochastic_process import RenewalProcess

        if isinstance(self.process, RenewalProcess):
            return RenewalProcessIterator(
                process=self.process,
                tf=self.tf,
                t0=self.t0,
                nb_samples=self.nb_samples,
                seed=self.seed,
            )
        else:
            return RenewalRewardProcessIterator(
                process=self.process,
                nb_samples=self.nb_samples,
                tf=self.tf,
                t0=self.t0,
                seed=self.seed,
            )


class NonHomogeneousPoissonProcessIterable(StochasticDataIterable):
    def __init__(
        self,
        process,
        nb_samples: int,
        tf: float,
        t0: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(nb_samples, tf, t0=t0, seed=seed)
        self.process = process

    def __iter__(self) -> NonHomogeneousPoissonProcessIterator:
        return NonHomogeneousPoissonProcessIterator(
            process=self.process,
            nb_samples=self.nb_samples,
            tf=self.tf,
            t0=self.t0,
            seed=self.seed,
        )
