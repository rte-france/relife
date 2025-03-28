from typing import Any, Optional, Sequence, Union, NewType, Generic, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

from relife.data import CountData
from relife.distributions.protocols import FittableLifetimeDistribution
from relife.likelihoods.mle import FittingResults, maximum_likelihood_estimation
from relife.plots import PlotConstructor, PlotNHPP

Z = TypeVarTuple("Z")
T = NewType("T", NDArray[np.floating] | NDArray[np.integer] | float | int)


# generic function
def nhpp_data_factory(
    events_assets_ids: Union[Sequence[str], NDArray[np.int64]],
    ages: NDArray[np.float64],
    /,
    *z: *Z,
    assets_ids: Optional[Union[Sequence[str], NDArray[np.int64]]] = None,
    first_ages: Optional[NDArray[np.float64]] = None,
    last_ages: Optional[NDArray[np.float64]] = None,
):
    # convert inputs to arrays
    events_assets_ids = np.asarray(events_assets_ids)
    ages = np.asarray(ages, dtype=np.float64)
    if assets_ids is not None:
        assets_ids = np.asarray(assets_ids)
    if first_ages is not None:
        first_ages = np.asarray(first_ages, dtype=np.float64)
    if last_ages is not None:
        last_ages = np.asarray(last_ages, dtype=np.float64)

    # control shapes
    if events_assets_ids.ndim != 1:
        raise ValueError("Invalid array shape for events_assets_ids. Expected 1d-array")
    if ages.ndim != 1:
        raise ValueError("Invalid array shape for ages_at_event. Expected 1d-array")
    if len(events_assets_ids) != len(ages):
        raise ValueError(
            "Shape of events_assets_ids and ages_at_event must be equal. Expected equal length 1d-arrays"
        )
    if assets_ids is not None:
        if assets_ids.ndim != 1:
            raise ValueError("Invalid array shape for assets_ids. Expected 1d-array")
        if first_ages is not None:
            if first_ages.ndim != 1:
                raise ValueError(
                    "Invalid array shape for start_ages. Expected 1d-array"
                )
            if len(first_ages) != len(assets_ids):
                raise ValueError(
                    "Shape of assets_ids and start_ages must be equal. Expected equal length 1d-arrays"
                )
        if last_ages is not None:
            if last_ages.ndim != 1:
                raise ValueError("Invalid array shape for end_ages. Expected 1d-array")
            if len(last_ages) != len(assets_ids):
                raise ValueError(
                    "Shape of assets_ids and end_ages must be equal. Expected equal length 1d-arrays"
                )
        if bool(z):
            for arg in z:
                arg = np.atleast_2d(np.asarray(arg, dtype=np.float64))
                if arg.ndim > 2:
                    raise ValueError(
                        "Invalid arg shape in model_args. Arrays must be 0, 1 or 2d"
                    )
                try:
                    arg.reshape((len(assets_ids), -1))
                except ValueError:
                    raise ValueError(
                        "Invalid arg shape in model_args. Arrays must coherent with the number of assets given by assets_ids"
                    )
    else:
        if first_ages is not None:
            raise ValueError(
                "If start_ages is given, corresponding asset ids must be given in assets_ids"
            )
        if last_ages is not None:
            raise ValueError(
                "If end_ages is given, corresponding asset ids must be given in assets_ids"
            )
        if bool(z):
            raise ValueError(
                "If model_args is given, corresponding asset ids must be given in assets_ids"
            )

    if events_assets_ids.dtype != np.int64:
        events_assets_ids = np.unique(events_assets_ids, return_inverse=True)[1]
    # convert assets_id to int id
    if assets_ids is not None:
        if assets_ids.dtype != np.int64:
            assets_ids = np.unique(assets_ids, return_inverse=True)[1]
        # control ids correspondance
        if not np.all(np.isin(events_assets_ids, assets_ids)):
            raise ValueError(
                "If assets_ids is filled, all values of events_assets_ids must exist in assets_ids"
            )

    # sort fields
    sort_ind = np.lexsort((ages, events_assets_ids))
    events_assets_ids = events_assets_ids[sort_ind]
    ages = ages[sort_ind]

    # number of age value per asset id
    nb_ages_per_asset = np.unique_counts(events_assets_ids).counts
    # index of the first ages and last ages in ages_at_event
    first_age_index = np.where(np.roll(events_assets_ids, 1) != events_assets_ids)[0]
    last_age_index = np.append(first_age_index[1:] - 1, len(events_assets_ids) - 1)

    if assets_ids is not None:

        # sort fields
        sort_ind = np.sort(assets_ids)
        first_ages = first_ages[sort_ind] if first_ages is not None else first_ages
        last_ages = last_ages[sort_ind] if last_ages is not None else last_ages
        z = tuple((arg[sort_ind] for arg in z))

        if first_ages is not None:
            if np.any(ages[first_age_index] <= first_ages[nb_ages_per_asset != 0]):
                raise ValueError(
                    "Each start_ages value must be lower than all of its corresponding ages_at_event values"
                )
        if last_ages is not None:
            if np.any(ages[last_age_index] >= last_ages[nb_ages_per_asset != 0]):
                raise ValueError(
                    "Each end_ages value must be greater than all of its corresponding ages_at_event values"
                )

    event = np.ones_like(ages, dtype=np.bool_)
    # insert_index = np.cumsum(nb_ages_per_asset)
    # insert_index = last_age_index + 1
    if last_ages is not None:
        time = np.insert(ages, last_age_index + 1, last_ages)
        event = np.insert(event, last_age_index + 1, False)
        _ids = np.insert(events_assets_ids, last_age_index + 1, assets_ids)
        if first_ages is not None:
            entry = np.insert(
                ages, np.insert((last_age_index + 1)[:-1], 0, 0), first_ages
            )
        else:
            entry = np.insert(ages, first_age_index, 0.0)
    else:
        time = ages.copy()
        _ids = events_assets_ids.copy()
        if first_ages is not None:
            entry = np.roll(ages, 1)
            entry[first_age_index] = first_ages
        else:
            entry = np.roll(ages, 1)
            entry[first_age_index] = 0.0
    model_args = tuple((np.take(arg, _ids) for arg in z))

    return time, event, entry, model_args


class NonHomogeneousPoissonProcess(Generic[*Z]):

    def __init__(
        self,
        distribution: FittableLifetimeDistribution[*Z],
    ):
        if not isinstance(distribution, FittableLifetimeDistribution):
            raise ValueError(
                "Invalid distribution : must be FittableLifetimeDistribution object."
            )
        self.distribution = distribution

    def intensity(self, time: T, *z: *Z) -> NDArray[np.float64]:
        return self.distribution.hf(time, *z)

    def cumulative_intensity(self, time: T, *z: *Z) -> NDArray[np.float64]:
        return self.distribution.chf(time, *z)

    def sample(
        self,
        size: int,
        tf: float,
        *z: *Z,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> CountData:
        from relife.sampling import sample_count_data

        return sample_count_data(
            self.distribution.freeze_zvariables(*z),
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
        *z: *Z,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> tuple[NDArray[np.float64], ...]:
        from relife.sampling import failure_data_sample

        return failure_data_sample(
            self.distribution.freeze_zvariables(*z),
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

    def fit(
        self,
        events_assets_ids: Union[Sequence[str], NDArray[np.int64]],
        events_ages: NDArray[np.float64],
        /,
        *z: *Z,
        assets_ids: Optional[Union[Sequence[str], NDArray[np.int64]]] = None,
        first_ages: Optional[NDArray[np.float64]] = None,
        last_ages: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> FittingResults:

        time, event, entry, model_args = nhpp_data_factory(
            events_assets_ids,
            events_ages,
            *z,
            assets_ids=assets_ids,
            first_ages=first_ages,
            last_ages=last_ages,
        )
        fitting_results = maximum_likelihood_estimation(
            self.distribution, time, *model_args, event=event, entry=entry, **kwargs
        )
        self.distribution.params = fitting_results.params
        return fitting_results


# class NonHomogeneousPoissonProcessWithRewards(NonHomogeneousPoissonProcess):
#     def __init__(
#         self,
#         model: LifetimeModel[*tuple[NumericalArrayLike, ...]],
#         rewards: Rewards,
#         model_args: tuple[NumericalArrayLike, ...] = (),
#         *,
#         discounting_rate: Optional[float] = None,
#         nb_assets: int = 1,
#     ):
#         super().__init__(model, model_args, nb_assets=nb_assets)
#         self.rewards = rewards
#         self.discounting = exp_discounting(discounting_rate)
