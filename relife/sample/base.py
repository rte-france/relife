from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Optional, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

from ..policy.base import RenewalPolicy
from ..stochastic_process import NonHomogeneousPoissonProcess, RenewalProcess
from ._dispatch_sample import failure_data_sample, sample_count_data

if TYPE_CHECKING:
    from .counting_data import CountData

Args = TypeVarTuple("Args")


class SampleMixin(Generic[*Args]):
    def sample(
        self: NonHomogeneousPoissonProcess[*Args] | RenewalProcess | RenewalPolicy,
        size: int,
        tf: float,
        /,
        *args: *Args,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> CountData:
        match self:
            case NonHomogeneousPoissonProcess():
                model = self.baseline.freeze(*args)
            case (RenewalProcess(), RenewalPolicy()):
                model = self
            case _:
                raise ValueError
        return sample_count_data(
            model,
            size,
            tf,
            t0=t0,
            maxsample=maxsample,
            seed=seed,
        )


class SampleFailureDataMixin(Generic[*Args]):
    def failure_data_sample(
        self: NonHomogeneousPoissonProcess[*Args] | RenewalProcess | RenewalPolicy,
        size: int,
        tf: float,
        /,
        *args: *Args,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> tuple[NDArray[np.float64], ...]:

        match self:
            case NonHomogeneousPoissonProcess():
                model = self.baseline.freeze(*args)
            case (RenewalProcess(), RenewalPolicy()):
                model = self
            case _:
                raise ValueError
        return failure_data_sample(
            model,
            size,
            tf,
            t0,
            maxsample=maxsample,
            seed=seed,
            use="model",
        )
