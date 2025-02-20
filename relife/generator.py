from typing import Iterator, Optional

import numpy as np
from numpy.typing import NDArray

from relife.core.discounting import exponential_discounting
from relife.core.nested_model import AgeReplacementModel
from relife.core.model import LifetimeModel
from relife.models import Exponential
from relife.types import (
    TupleArrays,
    Reward,
)


def rvs_size(
    nb_samples: int,
    nb_assets: int,
    model_args: TupleArrays = (),
):
    if bool(model_args) and model_args[0].ndim == 2:
        size = nb_samples  # rvs size
    else:
        size = nb_samples * nb_assets
    return size


def compute_events(
    lifetimes: NDArray[np.float64],
    model: LifetimeModel[*TupleArrays],
    model_args: TupleArrays = (),
) -> NDArray[np.bool_]:
    """
    tag lifetimes as being right censored or not depending on model used
    """
    events = np.ones_like(lifetimes, dtype=np.bool_)
    ar = 0.0
    if isinstance(model, AgeReplacementModel):
        ar = model_args[0]
    events[lifetimes < ar] = False
    return events


def lifetimes_generator(
    model: LifetimeModel[*TupleArrays],
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: TupleArrays = (),
    model1: Optional[LifetimeModel[*TupleArrays]] = None,
    model1_args: TupleArrays = (),
    seed: Optional[int] = None,
) -> Iterator[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
]:
    """Generates asset lifetimes data by sampling from given lifetime models."""

    event_times = np.zeros((nb_assets, nb_samples))
    still_valid = event_times < end_time

    def sample_routine(lifetime_model, args):
        nonlocal event_times, still_valid, seed  # modify these variables
        lifetimes = lifetime_model.rvs(
            *args,
            size=size,
            seed=seed,
        ).reshape((nb_assets, nb_samples))
        event_times += lifetimes
        events = compute_events(lifetimes, lifetime_model, args)
        still_valid = event_times < end_time
        assets_ids, samples_ids = np.where(still_valid)

        # update seed to avoid having the same rvs result
        if seed is not None:
            seed += 1

        return (
            samples_ids,
            assets_ids,
            lifetimes[still_valid],
            event_times[still_valid],
            events[still_valid],
            still_valid[still_valid],
        )

    if model1:
        size = rvs_size(nb_samples, nb_assets, model1_args)
        gen_data = sample_routine(model1, model1_args)
        if np.any(gen_data[-1]) > 0:
            yield gen_data
        else:
            return

    size = rvs_size(nb_samples, nb_assets, model_args)
    while True:
        gen_data = sample_routine(model, model_args)
        if np.any(gen_data[-1]) > 0:
            yield gen_data
        else:
            break
    return


def lifetimes_rewards_generator(
    model: LifetimeModel[*TupleArrays],
    reward: Reward,
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: TupleArrays = (),
    reward_args: TupleArrays = (),
    discounting_rate: float = 0.0,
    model1: Optional[LifetimeModel[*TupleArrays]] = None,
    model1_args: TupleArrays = (),
    reward1: Optional[Reward] = None,
    reward1_args: TupleArrays = (),
    seed: Optional[int] = None,
) -> Iterator[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
]:
    """
    Generates lifetimes and rewards until a specified end time for multiple assets and samples.

    Parameters:
    model: LifetimeModel
        Lifetime model used to simulate data.
    reward: Reward
        Callable used to compute rewards at each event times. This callable expects a `np.ndarray` as `timeline` followed
        by variable number of costs represented in `np.ndarray`. It returns one `np.ndarray`
    nb_samples: int
        Number of sample to generate for each asset.
    nb_assets: int
        Number of assets to generates data for.
    end_time: float
        Horizon time beyond which events are no longer simulated.
    model_args: ModelArgs, optional
        Any other variable values needed to compute model's functions.
    reward_args: TupleArrays, optional
        Other arguments required by `Reward` different from `timeline`.
    model1_args: ModelArgs, optional
        Any other variable values needed to compute model's functions.
    reward1_args: TupleArrays, optional
        Other arguments required by `Reward` different from `timeline`.
    seed: int, optional
        Optional seed for random number generation to ensure reproducibility.

    Returns:
    Iterator of tuple
        tuple of samples_ids, assets_ids, durations and nb_repairs
    """

    event_times = np.zeros((nb_assets, nb_samples))
    still_valid = event_times < end_time
    total_rewards = np.zeros((nb_assets, nb_samples))

    def sample_routine(
        lifetime_model, reward_func, lifetime_model_args, reward_func_args
    ):
        nonlocal event_times, still_valid, total_rewards, seed  # modify these variables
        lifetimes = lifetime_model.rvs(
            *lifetime_model_args,
            size=size,
            seed=seed,
        ).reshape((nb_assets, nb_samples))
        event_times += lifetimes
        events = compute_events(lifetimes, lifetime_model, lifetime_model_args)
        rewards = reward_func(lifetimes, *reward_func_args)
        discountings = exponential_discounting.factor(event_times, discounting_rate)
        total_rewards += rewards * discountings
        still_valid = event_times < end_time
        assets_ids, samples_ids = np.where(still_valid)

        # update seed to avoid having the same rvs result
        if seed is not None:
            seed += 1

        return (
            samples_ids,
            assets_ids,
            lifetimes[still_valid],
            event_times[still_valid],
            total_rewards[still_valid],
            events[still_valid],
        )

    if reward1 is None:
        reward1 = reward
        reward1_args = reward_args

    if model1:
        size = rvs_size(nb_samples, nb_assets, model1_args)
        gen_data = sample_routine(model1, reward1, model1_args, reward1_args)
        if gen_data[0].size > 0:
            yield gen_data
        else:
            return

    size = rvs_size(nb_samples, nb_assets, model_args)
    while True:
        gen_data = sample_routine(model, reward, model_args, reward_args)
        if gen_data[0].size > 0:
            yield gen_data
        else:
            break
    return


def nhpp_generator(
    model: LifetimeModel[*TupleArrays],
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: TupleArrays = (),
    seed: Optional[int] = None,
) -> Iterator[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ]
]:
    r"""
    Generates repairs events until a specified end time for multiple assets and samples.

    Parameters:
    model: LifetimeModel
        NHPP-based lifetime model used to simulate replacement policies.
    nb_samples: int
        Number of sample to generate for each asset.
    nb_assets: int
        Number of assets to generates data for.
    end_time: float
        Horizon time beyond which events are no longer simulated.
    model_args: ModelArgs, optional
        Any other variable values needed to compute model's functions.
    seed: int, optional
        Optional seed for random number generation to ensure reproducibility.

    Returns:
    Iterator of tuple
        tuple of samples_ids, assets_ids, durations and nb_repairs
    """
    hpp_timeline = np.zeros((nb_assets, nb_samples))
    previous_event_times = np.zeros((nb_assets, nb_samples))
    still_valid = np.ones_like(hpp_timeline, dtype=np.bool_)
    exponential_dist = Exponential(1.0)

    def sample_routine(target_model, args):
        nonlocal hpp_timeline, previous_event_times, still_valid, seed  # modify these variables
        hpp_timeline += exponential_dist.rvs(size=size, seed=seed).reshape(
            (nb_assets, nb_samples)
        )
        event_times = target_model.ichf(hpp_timeline, *args)
        durations = event_times - previous_event_times
        previous_event_times = event_times
        still_valid = event_times < end_time
        if seed is not None:
            seed += 1
        assets_ids, samples_ids = np.where(still_valid)
        return samples_ids, assets_ids, durations[still_valid], event_times[still_valid]

    size = rvs_size(nb_samples, nb_assets, model_args)
    while True:
        gen_data = sample_routine(model, model_args)
        if gen_data[0].size > 0:
            yield gen_data
        else:
            break
    return


def nhpp_policy_generator(
    model: LifetimeModel[*TupleArrays],
    ar: NDArray[np.float64],  # ages of replacement
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: TupleArrays = (),
    seed: Optional[int] = None,
) -> Iterator[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ]
]:
    r"""
    Generates repairs and replacement events until a specified end time for multiple assets and samples.

    Parameters:
    model: LifetimeModel
        NHPP-based lifetime model used to simulate replacement policies.
    ar: np.ndarray
        Replacement ages.
    nb_samples: int
        Number of sample to generate for each asset.
    nb_assets: int
        Number of assets to generates data for.
    end_time: float
        Horizon time beyond which events are no longer simulated.
    model_args: ModelArgs, optional
        Any other variable values needed to compute model's functions.
    seed: int, optional
        Optional seed for random number generation to ensure reproducibility.

    Returns:
    Iterator of tuple
        tuple of samples_ids, assets_ids, durations and nb_repairs
    """
    hpp_timeline = np.zeros((nb_assets, nb_samples))
    previous_event_times = np.zeros((nb_assets, nb_samples))
    timeline = np.zeros((nb_assets, nb_samples))
    nb_repairs = np.zeros((nb_assets, nb_samples), dtype=np.int64)
    still_valid = np.ones_like(hpp_timeline, dtype=np.bool_)
    exponential_dist = Exponential(1.0)
    ar = np.broadcast_to(ar.reshape(-1, 1), (nb_assets, nb_samples))

    def sample_routine(lifetime_model, args):
        nonlocal hpp_timeline, previous_event_times, timeline, nb_repairs, still_valid, seed  # modify these variables

        hpp_timeline += exponential_dist.rvs(size=size, seed=seed).reshape(
            (nb_assets, nb_samples)
        )
        event_times = lifetime_model.ichf(hpp_timeline, *args)  # ar values or less
        durations = event_times - previous_event_times

        # update
        still_repaired = event_times < ar
        nb_repairs[still_repaired] += 1
        durations[~still_repaired] = (
            ar[~still_repaired] - previous_event_times[~still_repaired]
        )

        timeline[~still_repaired] += (
            ar[~still_repaired] - previous_event_times[~still_repaired]
        )
        timeline[still_repaired] += durations[still_repaired]

        # milestones[~still_repaired] += ar[~still_repaired]
        hpp_timeline[~still_repaired] = 0.0
        nb_repairs[~still_repaired] = 0

        previous_event_times[still_repaired] = event_times[still_repaired]
        previous_event_times[~still_repaired] = 0.0

        assets_ids, samples_ids = np.where(still_valid)
        still_valid = timeline < end_time

        if seed is not None:
            seed += 1

        return (
            samples_ids,
            assets_ids,
            durations[still_valid],
            nb_repairs[still_valid],
        )

    size = rvs_size(nb_samples, nb_assets, model_args)
    while True:
        gen_data = sample_routine(model, model_args)
        if np.any(gen_data[-1]) > 0:
            yield gen_data
        else:
            break
    return
