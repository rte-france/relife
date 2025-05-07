from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    Self,
    Sequence,
    TypeVarTuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from relife._base import ParametricModel
from relife._plots import PlotNHPP

from .frozen_process import FrozenNonHomogeneousPoissonProcess

if TYPE_CHECKING:
    from relife.lifetime_model import FittableParametricLifetimeModel
    from relife.likelihood.maximum_likelihood_estimation import FittingResults
    from relife.sample import CountData

Args = TypeVarTuple("Args")


class NonHomogeneousPoissonProcess(ParametricModel, Generic[*Args]):

    def __init__(
        self,
        baseline: FittableParametricLifetimeModel[*Args],
    ):
        super().__init__()
        self.baseline = baseline

    @property
    def fitting_results(self) -> Optional[FittingResults]:
        return self._fitting_results

    @fitting_results.setter
    def fitting_results(self, value: FittingResults):
        self._fitting_results = value

    def intensity(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        return self.baseline.hf(time, *args)

    def cumulative_intensity(
        self, time: float | NDArray[np.float64], *args: *Args
    ) -> NDArray[np.float64]:
        return self.baseline.chf(time, *args)

    def sample(
        self,
        size: int,
        tf: float,
        /,
        *args: *Args,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> CountData:
        return self.freeze(*args).sample(size, tf, t0=t0, maxsample=maxsample, seed=seed)

    def failure_data_sample(
        self,
        size: int,
        tf: float,
        /,
        *args: *Args,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> tuple[NDArray[np.float64], ...]:
        return self.freeze(*args).failure_data_sample(size, tf, t0=t0, maxsample=maxsample, seed=seed)

    def freeze(self, *args: *Args) -> FrozenNonHomogeneousPoissonProcess:
        from .frozen_process import FrozenNonHomogeneousPoissonProcess

        args_names = self.baseline.args_names
        if len(args) != len(args_names):
            raise ValueError(
                f"Expected {args_names} positional arguments but got only {len(args)} arguments"
            )
        frozen_model = FrozenNonHomogeneousPoissonProcess(self)
        frozen_model.freeze_args(**{k: v for (k, v) in zip(args_names, args)})
        return frozen_model

    @property
    def plot(self) -> PlotNHPP:
        return PlotNHPP(self)

    def fit(
        self,
        events_assets_ids: Union[Sequence[str], NDArray[np.int64]],
        events_ages: NDArray[np.float64],
        /,
        *args: *Args,
        assets_ids: Optional[Union[Sequence[str], NDArray[np.int64]]] = None,
        first_ages: Optional[NDArray[np.float64]] = None,
        last_ages: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> Self:
        from relife.data import nhpp_data_factory
        from relife.likelihood import maximum_likelihood_estimation

        nhpp_data = nhpp_data_factory(
            events_assets_ids,
            events_ages,
            *args,
            assets_ids=assets_ids,
            first_ages=first_ages,
            last_ages=last_ages,
        )
        fitted_model = maximum_likelihood_estimation(self, nhpp_data, **kwargs)
        self.params = fitted_model.params
        return fitted_model
