from dataclasses import dataclass
from typing import Any, Iterator, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from relife import Exponential
from relife.likelihoods import LikelihoodFromLifetimes
from relife.model import (
    LifetimeModel,
    ParametricLifetimeModel,
)
from relife.sampling import model_rvs, rvs_size
from relife.utils.data import CountData, LifetimeData, lifetime_data_factory
from relife.utils.types import ModelArgs, VariadicArgs


class StochasticProcess(Protocol):
    model: LifetimeModel[*ModelArgs]

    def sample(self, nb_sample: int) -> CountData: ...


# DATA FORMAT
# t0 : ages at the beginning of the observation window
# tf : ages at the end of the observation window
# ages : ages at each failure
# assets : corresponding asset indices


def nhpp_data_factory(
    ages: NDArray[np.float64],
    assets: NDArray[np.int64],
) -> LifetimeData:
    """
    >>> assets = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2])
    >>> ages = np.array([1, 2, 4, 5, 3, 8, 6, 12, 15])

    >>> permutation_ind = np.random.permutation(len(assets))
    >>> ages = ages[permutation_ind]
    >>> assets = assets[permutation_ind]
    """

    # insert t0 and tf at the beginning and end
    #
    # sort_ind = np.lexsort((ages, assets))
    # ages = ages[sort_ind]
    # assets = assets[sort_ind]
    # ages = np.insert(ages, np.cumsum(np.unique(assets), tf)
    # ages = np.insert(ages, np.insert(np.cumsum(np.unique(assets)[:-1]), 0, 0), t0)

    sort_ind = np.lexsort((ages, assets))
    changing_asset_ind = np.where(assets[sort_ind][:-1] != assets[sort_ind][1:])[0]

    time = ages[sort_ind][1:]
    entry = ages[sort_ind][:-1]
    assets = assets[sort_ind][1:]

    event = np.ones_like(time, dtype=np.bool_)
    last_ind_per_asset = np.append(
        np.where(assets[:-1] != assets[1:])[0], len(assets) - 1
    )
    event[last_ind_per_asset] = False

    entry = np.delete(entry, changing_asset_ind)
    time = np.delete(time, changing_asset_ind)
    event = np.delete(event, changing_asset_ind)

    return time, event, entry


@dataclass
class NHPPData(CountData):
    # durations in post_init ?

    def number_of_repairs(self):
        pass

    def mean_number_of_repairs(self):
        pass

    def to_nhpp_data(self):
        pass


# TODO : pass it as ParametricModel to compose_with and access params
class NHPP:

    def __init__(self, model: ParametricLifetimeModel[*ModelArgs]):
        self.model = model

    def intensity(self, time: np.ndarray, *args: *ModelArgs) -> np.ndarray:
        return self.model.hf(time, *args)

    def cumulative_intensity(self, time: np.ndarray, *args: *ModelArgs) -> np.ndarray:
        return self.model.chf(time, *args)

    def sample(self, nb_sample: int, end_time: float) -> NHPPData:
        pass

    def fit(
        self,
        ages: NDArray[np.float64],
        assets: NDArray[np.float64],
        model_args: tuple[*VariadicArgs] = (),
        inplace: bool = True,
        **kwargs: Any,
    ) -> NDArray[np.float64]:

        lifetime_data = lifetime_data_factory(nhpp_data_factory(ages, assets))

        optimized_model = self.model.copy()
        optimized_model.init_params(lifetime_data, *model_args)
        # or just optimized_model.init_params(observed_lifetimes, *model_args)

        likelihood = LikelihoodFromLifetimes(
            optimized_model, lifetime_data, model_args=model_args
        )

        minimize_kwargs = {
            "method": kwargs.get("method", "L-BFGS-B"),
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
            jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )

        if inplace:
            self.model.init_params(lifetime_data, *model_args)
            # or just self.init_params(observed_lifetimes, *model_args)
            self.model.params = likelihood.params

        return optimizer.x


def nhpp_generator(
    model: LifetimeModel[*ModelArgs],
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: ModelArgs = (),
) -> Iterator[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ]
]:
    hpp_event_times = np.zeros((nb_assets, nb_samples))
    still_valid = np.ones_like(hpp_event_times, dtype=np.bool_)
    exponential_dist = Exponential(1.0)

    def sample_routine(target_model, args):
        nonlocal hpp_event_times, still_valid  # modify these variables
        lifetimes = model_rvs(exponential_dist, size, args=args).reshape(
            (nb_assets, nb_samples)
        )
        hpp_event_times += lifetimes
        event_times = target_model.ichf(hpp_event_times, *args)
        still_valid = event_times < end_time
        return event_times, still_valid

    size = rvs_size(nb_samples, nb_assets, model_args)
    while True:
        gen_data = sample_routine(model, model_args)
        if np.any(gen_data[-1]) > 0:
            yield gen_data
        else:
            break
    return
