from typing import Any, Optional, Self, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from relife.core.descriptors import ShapedArgs
from relife.core.likelihoods import LikelihoodFromLifetimes
from relife.core.model import LifetimeModel, ParametricModel
from relife.data import CountData, lifetime_data_factory
from relife.plots import PlotConstructor, PlotNHPP
from relife.rewards import RewardsFunc, exp_discounting
from relife.types import Args, VariadicArgs


def nhpp_data_factory(
    events_assets_ids: Union[Sequence[str], NDArray[np.int64]],
    ages_at_events: NDArray[np.float64],
    assets_ids: Optional[Union[Sequence[str], NDArray[np.int64]]] = None,
    start_ages: Optional[NDArray[np.float64]] = None,
    end_ages: Optional[NDArray[np.float64]] = None,
    model_args: tuple[Args, ...] = (),
):
    # convert inputs to arrays
    events_assets_ids = np.asarray(events_assets_ids)
    ages_at_events = np.asarray(ages_at_events, dtype=np.float64)
    if assets_ids is not None:
        assets_ids = np.asarray(assets_ids)
    if start_ages is not None:
        start_ages = np.asarray(start_ages, dtype=np.float64)
    if end_ages is not None:
        end_ages = np.asarray(end_ages, dtype=np.float64)

    # control shapes
    if events_assets_ids.ndim != 1:
        raise ValueError("Invalid array shape for events_assets_ids. Expected 1d-array")
    if ages_at_events.ndim != 1:
        raise ValueError("Invalid array shape for ages_at_event. Expected 1d-array")
    if len(events_assets_ids) != len(ages_at_events):
        raise ValueError(
            "Shape of events_assets_ids and ages_at_event must be equal. Expected equal length 1d-arrays"
        )
    if assets_ids is not None:
        if assets_ids.ndim != 1:
            raise ValueError("Invalid array shape for assets_ids. Expected 1d-array")
        if start_ages is not None:
            if start_ages.ndim != 1:
                raise ValueError(
                    "Invalid array shape for start_ages. Expected 1d-array"
                )
            if len(start_ages) != len(assets_ids):
                raise ValueError(
                    "Shape of assets_ids and start_ages must be equal. Expected equal length 1d-arrays"
                )
        if end_ages is not None:
            if end_ages.ndim != 1:
                raise ValueError("Invalid array shape for end_ages. Expected 1d-array")
            if len(end_ages) != len(assets_ids):
                raise ValueError(
                    "Shape of assets_ids and end_ages must be equal. Expected equal length 1d-arrays"
                )
        if bool(model_args):
            for arg in model_args:
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
        if start_ages is not None:
            raise ValueError(
                "If start_ages is given, corresponding asset ids must be given in assets_ids"
            )
        if end_ages is not None:
            raise ValueError(
                "If end_ages is given, corresponding asset ids must be given in assets_ids"
            )
        if bool(model_args):
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
    sort_ind = np.lexsort((ages_at_events, events_assets_ids))
    events_assets_ids = events_assets_ids[sort_ind]
    ages_at_events = ages_at_events[sort_ind]

    # number of age value per asset id
    nb_ages_per_asset = np.unique_counts(events_assets_ids).counts
    # index of the first ages and last ages in ages_at_event
    first_age_index = np.where(np.roll(events_assets_ids, 1) != events_assets_ids)[0]
    last_age_index = np.append(first_age_index[1:] - 1, len(events_assets_ids) - 1)

    if assets_ids is not None:

        # sort fields
        sort_ind = np.sort(assets_ids)
        start_ages = start_ages[sort_ind] if start_ages is not None else start_ages
        end_ages = end_ages[sort_ind] if end_ages is not None else end_ages
        model_args = tuple((arg[sort_ind] for arg in model_args))

        if start_ages is not None:
            if np.any(
                ages_at_events[first_age_index] <= start_ages[nb_ages_per_asset != 0]
            ):
                raise ValueError(
                    "Each start_ages value must be lower than all of its corresponding ages_at_event values"
                )
        if end_ages is not None:
            if np.any(
                ages_at_events[last_age_index] >= end_ages[nb_ages_per_asset != 0]
            ):
                raise ValueError(
                    "Each end_ages value must be greater than all of its corresponding ages_at_event values"
                )

    event = np.ones_like(ages_at_events, dtype=np.bool_)
    # insert_index = np.cumsum(nb_ages_per_asset)
    # insert_index = last_age_index + 1
    if end_ages is not None:
        time = np.insert(ages_at_events, last_age_index + 1, end_ages)
        event = np.insert(event, last_age_index + 1, False)
        _ids = np.insert(events_assets_ids, last_age_index + 1, assets_ids)
        if start_ages is not None:
            entry = np.insert(
                ages_at_events, np.insert((last_age_index + 1)[:-1], 0, 0), start_ages
            )
        else:
            entry = np.insert(ages_at_events, first_age_index, 0.0)
    else:
        time = ages_at_events.copy()
        _ids = events_assets_ids.copy()
        if start_ages is not None:
            entry = np.roll(ages_at_events, 1)
            entry[first_age_index] = start_ages
        else:
            entry = np.roll(ages_at_events, 1)
            entry[first_age_index] = 0.0
    model_args = tuple((np.take(arg, _ids) for arg in model_args))

    return time, event, entry, model_args


class NonHomogeneousPoissonProcess(ParametricModel):

    model_args = ShapedArgs(astuple=True)

    def __init__(
        self,
        model: LifetimeModel[*tuple[Args, ...]],
        model_args: tuple[Args, ...] = (),
        *,
        nb_assets: int = 1,
    ):
        super().__init__()
        self.nb_assets = nb_assets
        self.compose_with(model=model)
        self.model_args = model_args

    def intensity(self, time: np.ndarray, *args: *tuple[Args, ...]) -> np.ndarray:
        return self.model.hf(time, *args)

    def cumulative_intensity(
        self, time: np.ndarray, *args: *tuple[Args, ...]
    ) -> np.ndarray:
        return self.model.chf(time, *args)

    def sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> CountData:
        from relife.sampling import sample_count_data

        return sample_count_data(self, size, tf, t0=t0, maxsample=maxsample, seed=seed)

    def to_failure_data(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> tuple[NDArray[np.float64], ...]:
        from relife.sampling import sample_failure_data

        return sample_failure_data(
            self, size, tf, t0, maxsample=maxsample, seed=seed, use="model"
        )

    @property
    def plot(self) -> PlotConstructor:
        return PlotNHPP(self)

    def fit(
        self,
        a0: NDArray[np.float64],
        af: NDArray[np.float64],
        ages: NDArray[np.float64],
        assets: NDArray[np.int64],
        model_args: tuple[*VariadicArgs] = (),
        **kwargs: Any,
    ) -> Self:

        lifetime_data = lifetime_data_factory(*nhpp_data_factory(a0, af, ages, assets))

        optimized_model = self.model.copy()
        optimized_model.init_params(lifetime_data, *model_args)
        # or just optimized_model.init_params(observed_lifetimes, *model_args)

        likelihood = LikelihoodFromLifetimes(
            optimized_model, lifetime_data, model_args=model_args
        )

        minimize_kwargs = {
            "method": kwargs.get("method", "Nelder-Mead"),  # Nelder-Mead better here
            "constraints": kwargs.get("constraints", ()),
            "tol": kwargs.get("tol", None),
            "callback": kwargs.get("callback", None),
            "options": kwargs.get("options", None),
            "bounds": kwargs.get("bounds", optimized_model.params_bounds),
            "x0": kwargs.get("x0", optimized_model.params),
        }

        optimizer = minimize(
            likelihood.negative_log,
            minimize_kwargs.pop("x0"),
            # jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )
        optimized_model.params = optimizer.x

        self.model.init_params(lifetime_data, *model_args)
        self.model.params = optimized_model.params

        return self


class NonHomogeneousPoissonProcessWithRewards(NonHomogeneousPoissonProcess):
    def __init__(
        self,
        model: LifetimeModel[*tuple[Args, ...]],
        rewards: RewardsFunc,
        model_args: tuple[Args, ...] = (),
        *,
        discounting_rate: Optional[float] = None,
        nb_assets: int = 1,
    ):
        super().__init__(model, model_args, nb_assets=nb_assets)
        self.rewards = rewards
        self.discounting = exp_discounting(discounting_rate)
