from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

from relife import ParametricModel

if TYPE_CHECKING:
    from .non_homogeneous_poisson_process import NonHomogeneousPoissonProcess


Args = TypeVarTuple("Args")


class FrozenNonHomogeneousPoissonProcess(ParametricModel, Generic[*Args]):
    def __init__(self, baseline: NonHomogeneousPoissonProcess[*tuple[float | NDArray, ...]]):
        super().__init__()
        self.baseline = baseline

    def intensity(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.intensity(time, *self.args)

    def cumulative_intensity(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.cumulative_intensity(time, *self.args)

    def sample_nhhp_data(self):
        pass

    def sample_count_data(self):
        pass

    # @property
    # def plot(self) -> PlotConstructor:
    #     return PlotNHPP(self)
