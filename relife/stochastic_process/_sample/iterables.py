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
    CountDataIterator,
    NonHomogeneousPoissonProcessIterator,
    RenewalProcessIterator,
    RenewalRewardProcessIterator,
)

if TYPE_CHECKING:
    from relife.stochastic_process import RenewalProcess

Args = TypeVarTuple("Args")


class CountDataIterable(Iterable[NDArray[np.void]], ABC):
    def __init__(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.size = size
        self.tf = tf
        self.t0 = t0
        self.nb_assets = nb_assets
        self.seed = seed

    @abstractmethod
    @override
    def __iter__(self) -> CountDataIterator: ...


def age_of_renewal_process_sampler(
    lifetime_model,
    nb_samples: int,
    t: float,
    nb_assets: int = 1,
    first_lifetime_model: Optional = None,
    seed: Optional[int] = None,
):
    timeline = np.zeros((nb_assets, nb_samples), dtype=np.float64)
    just_crossed_t = np.zeros_like(timeline, dtype=np.uint32)
    age_process = np.zeros_like(timeline, dtype=np.float64)
    replacement_cycle = 0

    while np.any(timeline < t):
        if replacement_cycle == 0 and first_lifetime_model is not None:
            time, entry = first_lifetime_model.rvs((nb_assets, nb_samples), return_entry=True, seed=seed)
        else:
            time, entry = lifetime_model.rvs((nb_assets, nb_samples), return_entry=True, seed=seed)
        replacement_cycle += 1
        residual_time = time - entry
        timeline += residual_time
        just_crossed_t[timeline > t] += 1
        age_process = np.where(just_crossed_t == 1, time - (timeline - t), age_process)
        if seed is not None:
            seed += 1
    return np.squeeze(age_process)


class RenewalProcessIterable(CountDataIterable):

    def __init__(
        self,
        process,
        size: int,
        tf: float,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(size, tf, t0=t0, nb_assets=nb_assets, seed=seed)
        # TODO : control and broadcast size here !
        self.process = process

    def __iter__(self) -> RenewalProcessIterator:
        from relife.stochastic_process import RenewalProcess

        if isinstance(self.process, RenewalProcess):
            return RenewalProcessIterator(
                self.process, self.size, self.tf, t0=self.t0, nb_assets=self.nb_assets, seed=self.seed
            )
        else:
            return RenewalRewardProcessIterator(
                self.process, self.size, self.tf, t0=self.t0, nb_assets=self.nb_assets, seed=self.seed
            )


class NonHomogeneousPoissonProcessIterable(CountDataIterable):
    def __init__(
        self,
        process,
        size: int,
        tf: float,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(size, tf, t0=t0, nb_assets=nb_assets, seed=seed)
        # TODO : control and broadcast size here !
        self.process = process
        if nb_assets is None:
            self.nb_assets = getattr(process, "nb_assets", 1)

    def __iter__(self) -> NonHomogeneousPoissonProcessIterator:
        return NonHomogeneousPoissonProcessIterator(
            self.process, self.size, self.tf, t0=self.t0, nb_assets=self.nb_assets, seed=self.seed
        )
