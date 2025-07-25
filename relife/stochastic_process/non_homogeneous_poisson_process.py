from __future__ import annotations

from typing import (
    Any,
    Optional,
    Self,
    Sequence,
    Union,
    TypeVar,
    Generic,
)

import numpy as np
from numpy.typing import NDArray

from relife import ParametricModel
from relife.data import NHPPData
from relife.lifetime_model import (
    LifetimeDistribution,
    LifetimeRegression,
    LeftTruncatedModel,
    FrozenLifetimeRegression,
    FrozenLeftTruncatedModel,
)
from relife.likelihood import LikelihoodFromLifetimes, FittingResults

M = TypeVar(
    "M",
    bound=Union[
        LifetimeDistribution, LifetimeRegression, LeftTruncatedModel, FrozenLifetimeRegression, FrozenLeftTruncatedModel
    ],
)


class NonHomogeneousPoissonProcess(ParametricModel, Generic[M]):

    def __init__(self, baseline: M):
        super().__init__()
        self.baseline = baseline

    @property
    def fitting_results(self) -> FittingResults:
        return self.baseline.fitting_results

    @fitting_results.setter
    def fitting_results(self, value: FittingResults):
        self.baseline.fitting_results = value

    def intensity(self, time: float | NDArray[np.float64], *args: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.hf(time, *args)

    def cumulative_intensity(
        self, time: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.baseline.chf(time, *args)

    def fit(
        self,
        events_assets_ids: Union[Sequence[str], NDArray[np.int64]],
        ages_at_events: NDArray[np.float64],
        *args: float | NDArray[np.float64],
        assets_ids: Optional[Union[Sequence[str], NDArray[np.int64]]] = None,
        first_ages: Optional[NDArray[np.float64]] = None,
        last_ages: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> Self:
        nhpp_data = NHPPData(
            events_assets_ids,
            ages_at_events,
            *args,
            assets_ids=assets_ids,
            first_ages=first_ages,
            last_ages=last_ages,
        )
        lifetime_data = nhpp_data.to_lifetime_data()
        self.baseline._init_params(lifetime_data, *args)
        likelihood = LikelihoodFromLifetimes(self.baseline, lifetime_data)
        fitting_results = likelihood.maximum_likelihood_estimation(**kwargs)
        self.params = fitting_results.optimal_params
        self.fitting_results = fitting_results
        return self


class FrozenNonHomogeneousPoissonProcess(ParametricModel, Generic[M]):
    def __init__(self, model: NonHomogeneousPoissonProcess[M], args_nb_assets: int, *args: float | NDArray[np.float64]):
        super().__init__()
        self.unfrozen_model = model
        self.frozen_args = args
        self.args_nb_assets = args_nb_assets

    def unfreeze(self) -> NonHomogeneousPoissonProcess[M]:
        return self.unfrozen_model

    def intensity(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.unfrozen_model.intensity(time, *self.frozen_args)

    def cumulative_intensity(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.unfrozen_model.cumulative_intensity(time, *self.frozen_args)

    def sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        nb_assets : Optional[int] = None,
        seed: Optional[int] = None,
    ):
        from ._sample import NonHomogeneousPoissonProcessIterable, NonHomogeneousPoissonProcessSample

        iterable = NonHomogeneousPoissonProcessIterable(self, size, tf, t0=t0, nb_assets=nb_assets, seed=seed)
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(struct_array, order=("sample_id", "asset_id", "timeline"))
        return NonHomogeneousPoissonProcessSample(t0, tf, struct_array)

    def generate_failure_data(
        self,
        size : int,
        tf: float,
        t0: float = 0.0,
        nb_assets : Optional[int] = None,
        seed: Optional[int] = None,
    ):
        from ._sample import NonHomogeneousPoissonProcessIterable

        iterable = NonHomogeneousPoissonProcessIterable(self, size, tf, t0=t0, nb_assets=nb_assets, seed=seed)
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(struct_array, order=("sample_id", "asset_id", "timeline"))

        first_ages_index = np.nonzero(struct_array["entry"] == t0)
        last_ages_index = np.nonzero(struct_array["age"] == tf)

        event_index = np.nonzero(struct_array["event"])

        first_ages = struct_array[first_ages_index]["entry"].copy()
        last_ages = struct_array[last_ages_index]["age"].copy()

        assets_ids = np.char.add(
            np.char.add(
                np.full_like(struct_array[last_ages_index]["sample_id"], "S", dtype=np.str_),
                struct_array[last_ages_index]["sample_id"].astype(np.str_),
            ),
            np.char.add(
                np.full_like(struct_array[last_ages_index]["asset_id"], "A", dtype=np.str_),
                struct_array[last_ages_index]["asset_id"].astype(np.str_),
            ),
        )

        events_assets_ids = np.char.add(
            np.char.add(
                np.full_like(struct_array[event_index]["sample_id"], "S", dtype=np.str_),
                struct_array[event_index]["sample_id"].astype(np.str_),
            ),
            np.char.add(
                np.full_like(struct_array[event_index]["asset_id"], "A", dtype=np.str_),
                struct_array[event_index]["asset_id"].astype(np.str_),
            ),
        )
        ages_at_events = struct_array[event_index]["age"].copy()

        return {
            "events_assets_ids": events_assets_ids,
            "ages_at_events": ages_at_events,
            "assets_ids": assets_ids,
            "first_ages": first_ages,
            "last_ages": last_ages,
        }
