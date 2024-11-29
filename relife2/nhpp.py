from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from relife2 import Exponential, LifetimeModel, ParametricLifetimeModel
from relife2.data import CountData, LifetimeData, lifetime_data_factory
from relife2.fiability.likelihood import LikelihoodFromLifetimes
from relife2.renewal.reward import Reward
from relife2.types import ModelArgs, RewardArgs, VariadicArgs


class StochasticProcess(Protocol):
    model: LifetimeModel[*ModelArgs]

    def sample(self, nb_sample: int) -> CountData: ...


def _nhpp_data_factory_nhpp(
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

    return lifetime_data_factory(
        time,
        event,
        entry,
    )


# TODO : pass it as ParametricModel to compose_with and access params
class NHPP:

    def __init__(self, model: ParametricLifetimeModel[*ModelArgs]):
        self.model = model

    def fit(
        self,
        ages: NDArray[np.float64],
        assets: NDArray[np.float64],
        model_args: tuple[*VariadicArgs] = (),
        inplace: bool = True,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        lifetime_data = _nhpp_data_factory_nhpp(ages, assets)

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

    def sample(self, nb_sample: int) -> CountData:
        pass


def model_rvs(
    model: LifetimeModel[*ModelArgs],
    size: int,
    args: ModelArgs = (),
):
    return model.rvs(*args, size=size)


def rvs_size(
    nb_samples: int,
    nb_assets: int,
    model_args: ModelArgs = (),
):
    if bool(model_args) and model_args[0].ndim == 2:
        size = nb_samples  # rvs size
    else:
        size = nb_samples * nb_assets
    return size


def compute_rewards(
    reward: Reward[*RewardArgs],
    lifetimes: NDArray[np.float64],
    args: RewardArgs = (),
):
    return reward(lifetimes, *args)


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
        ages = np.diff(event_times, axis=0, prepend=0)
        still_valid = event_times < end_time
        return ages, event_times, still_valid

    size = rvs_size(nb_samples, nb_assets, model_args)
    while True:
        gen_data = sample_routine(model, model_args)
        if np.any(gen_data[-1]) > 0:
            yield gen_data
        else:
            break
    return


@dataclass
class NHPPData(CountData):
    ages: NDArray[np.float64] = field(repr=False)

    def to_nhpp_data(self):
        pass
