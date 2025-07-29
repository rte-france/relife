from __future__ import annotations

from typing import (
    Any,
    Optional,
    Self,
    Sequence,
    TypeVarTuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from relife.data import NHPPData
from relife.lifetime_model import FittableParametricLifetimeModel
from relife.likelihood import FittingResults, LikelihoodFromLifetimes

from .base import FrozenStochasticProcess, StochasticProcess

Args = TypeVarTuple("Args")


class NonHomogeneousPoissonProcess(StochasticProcess[*Args]):

    def __init__(self, lifetime_model: FittableParametricLifetimeModel[*Args]):
        super().__init__()
        self.lifetime_model = lifetime_model

    @property
    def fitting_results(self) -> FittingResults:
        return self.lifetime_model.fitting_results

    @fitting_results.setter
    def fitting_results(self, value: FittingResults):
        self.lifetime_model.fitting_results = value

    def intensity(self, time: float | NDArray[np.float64], *args: *Args) -> NDArray[np.float64]:
        return self.lifetime_model.hf(time, *args)

    def cumulative_intensity(self, time: float | NDArray[np.float64], *args: *Args) -> NDArray[np.float64]:
        return self.lifetime_model.chf(time, *args)

    def freeze(self, *args: *Args):
        return FrozenNonHomogeneousPoissonProcess(self, *args)

    def sample(
        self,
        size: int,
        tf: float,
        *args: *Args,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        seed: Optional[int] = None,
    ):

        from ._sample import (
            NonHomogeneousPoissonProcessIterable,
            NonHomogeneousPoissonProcessSample,
        )

        frozen_nhpp = self.freeze(*args)
        iterable = NonHomogeneousPoissonProcessIterable(frozen_nhpp, size, tf, t0=t0, nb_assets=nb_assets, seed=seed)
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(struct_array, order=("sample_id", "asset_id", "timeline"))
        return NonHomogeneousPoissonProcessSample(t0, tf, struct_array)

    def generate_failure_data(
        self,
        size: int,
        tf: float,
        *args: *Args,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        from ._sample import NonHomogeneousPoissonProcessIterable

        frozen_nhpp = self.freeze(*args)

        iterable = NonHomogeneousPoissonProcessIterable(frozen_nhpp, size, tf, t0=t0, nb_assets=nb_assets, seed=seed)
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

    def fit(
        self,
        events_assets_ids: Union[Sequence[str], NDArray[np.int64]],
        ages_at_events: NDArray[np.float64],
        *args: *Args,
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
        # noinspection PyProtectedMember
        self.lifetime_model._init_params(lifetime_data)
        likelihood = LikelihoodFromLifetimes(self.baseline, lifetime_data)
        fitting_results = likelihood.maximum_likelihood_estimation(**kwargs)
        self.params = fitting_results.optimal_params
        self.fitting_results = fitting_results
        return self


class FrozenNonHomogeneousPoissonProcess(FrozenStochasticProcess[*Args]):
    def __init__(self, model: NonHomogeneousPoissonProcess[*Args], *args: *Args):
        super().__init__(model, *args)

    def intensity(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.unfrozen_model.intensity(time, *self.args)

    def cumulative_intensity(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.unfrozen_model.cumulative_intensity(time, *self.args)

    def sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        return self.unfrozen_model.sample(size, tf, *self.args, t0=t0, nb_assets=nb_assets, seed=seed)

    def generate_failure_data(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        nb_assets: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        return self.unfrozen_model.generate_failure_data(
            size, tf, t0=t0, nb_assets=nb_assets, seed=seed, *self.args_values
        )
