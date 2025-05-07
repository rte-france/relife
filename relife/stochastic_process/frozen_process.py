from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from relife import ParametricModel
from relife._base import FrozenMixin
from relife._plots import PlotConstructor, PlotNHPP
from relife.sample import SampleMixin, FailureDataSampleMixin

if TYPE_CHECKING:
    from .non_homogeneous_poisson_process import NonHomogeneousPoissonProcess


class FrozenNonHomogeneousPoissonProcess(ParametricModel, FrozenMixin, SampleMixin, FailureDataSampleMixin):
    def __init__(
        self, baseline: NonHomogeneousPoissonProcess[*tuple[float | NDArray, ...]]
    ):
        super().__init__()
        self.baseline = baseline

    def intensity(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.intensity(time, *self.args)

    def cumulative_intensity(
        self, time: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.model.cumulative_intensity(time, *self.args)

    @property
    def plot(self) -> PlotConstructor:
        return PlotNHPP(self)
