from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from relife import FrozenParametricModel
from relife._plots import PlotConstructor, PlotNHPP
from relife.sample import CountData

from .nhpp import NonHomogeneousPoissonProcess


class FrozenNonHomogeneousPoissonProcess(FrozenParametricModel):
    model: NonHomogeneousPoissonProcess[*tuple[float | NDArray, ...]]

    def intensity(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.intensity(time, *self.args)

    def cumulative_intensity(
        self, time: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.model.cumulative_intensity(time, *self.args)

    def sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> CountData:
        from relife.sample import sample_count_data

        return sample_count_data(
            self.model,
            size,
            tf,
            t0=t0,
            maxsample=maxsample,
            seed=seed,
        )

    def failure_data_sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> tuple[NDArray[np.float64], ...]:
        from relife.sample import failure_data_sample

        return failure_data_sample(
            self.model,
            size,
            tf,
            t0,
            maxsample=maxsample,
            seed=seed,
            use="model",
        )

    @property
    def plot(self) -> PlotConstructor:
        return PlotNHPP(self)
