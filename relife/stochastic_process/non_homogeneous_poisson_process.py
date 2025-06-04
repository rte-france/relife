from __future__ import annotations

from typing import (
    Any,
    Optional,
    Self,
    Sequence,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from relife import ParametricModel
from relife.data import NHPPData
from relife.lifetime_model import LifetimeDistribution, LifetimeRegression


class NonHomogeneousPoissonProcess(ParametricModel):

    def __init__(self, baseline: LifetimeDistribution | LifetimeRegression):
        super().__init__()
        self.baseline = baseline

    @property
    def fitting_results(self):
        return self.baseline.fitting_results

    def intensity(self, time: float | NDArray[np.float64], *args: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.hf(time, *args)

    def cumulative_intensity(
        self, time: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.baseline.chf(time, *args)

    # def sample(
    #     self,
    #     size: int,
    #     tf: float,
    #     /,
    #     *args: float | NDArray[np.float64],
    #     t0: float = 0.0,
    #     maxsample: int = 1e5,
    #     seed: Optional[int] = None,
    # ):
    #
    # def failure_data_sample(
    #     self,
    #     size: int,
    #     tf: float,
    #     /,
    #     *args: float | NDArray[np.float64],
    #     t0: float = 0.0,
    #     maxsample: int = 1e5,
    #     seed: Optional[int] = None,
    # ) -> tuple[NDArray[np.float64], ...]:
    #     return self.freeze(*args).failure_data_sample(size, tf, t0=t0, maxsample=maxsample, seed=seed)

    # @property
    # def plot(self) -> PlotNHPP:
    #     return PlotNHPP(self)

    def fit(
        self,
        events_assets_ids: Union[Sequence[str], NDArray[np.int64]],
        events_ages: NDArray[np.float64],
        /,
        *args: float | NDArray[np.float64],
        assets_ids: Optional[Union[Sequence[str], NDArray[np.int64]]] = None,
        first_ages: Optional[NDArray[np.float64]] = None,
        last_ages: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> Self:
        nhpp_data = NHPPData(
            events_assets_ids,
            events_ages,
            *args,
            assets_ids=assets_ids,
            first_ages=first_ages,
            last_ages=last_ages,
        )
        lifetime_data = nhpp_data.to_lifetime_data()
        self.baseline = self.baseline._fit_from_lifetime_data(lifetime_data)
        return self


class FrozenNonHomogeneousPoissonProcess(ParametricModel):
    def __init__(self, model: NonHomogeneousPoissonProcess, args_nb_assets: int, *args: float | NDArray[np.float64]):
        super().__init__()
        self.unfrozen_model = model
        self.frozen_args = args
        self.args_nb_assets = args_nb_assets

    def unfreeze(self) -> NonHomogeneousPoissonProcess:
        return self.unfrozen_model

    def intensity(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.intensity(time, *self.frozen_args)

    def cumulative_intensity(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.cumulative_intensity(time, *self.frozen_args)

    def sample_nhhp_data(self):
        pass

    def sample_count_data(self):
        pass

    # @property
    # def plot(self) -> PlotConstructor:
    #     return PlotNHPP(self)
