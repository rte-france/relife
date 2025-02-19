from typing import Iterator, Optional

import numpy as np
from nbclient.exceptions import timeout_err_msg
from numpy.typing import NDArray

from relife.core.discounting import exponential_discounting
from relife.core.nested_model import AgeReplacementModel
from relife.core.model import LifetimeModel
from relife.models import Exponential
from relife.types import (
    Model1Args,
    ModelArgs,
    Reward1Args,
    RewardArgs,
    Reward,
)


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


def compute_events(
    lifetimes: NDArray[np.float64],
    model: LifetimeModel[*ModelArgs],
    model_args: ModelArgs = (),
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
    model: LifetimeModel[*ModelArgs],
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: ModelArgs = (),
    model1: Optional[LifetimeModel[*Model1Args]] = None,
    model1_args: Model1Args = (),
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
        # update seed to avoid having the same rvs result
        if seed is not None:
            seed += 1
        return lifetimes, event_times, events, still_valid

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
    model: LifetimeModel[*ModelArgs],
    reward: Reward,
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: ModelArgs = (),
    reward_args: RewardArgs = (),
    discounting_rate: float = 0.0,
    model1: Optional[LifetimeModel[*Model1Args]] = None,
    model1_args: Model1Args = (),
    reward1: Optional[Reward] = None,
    reward1_args: Reward1Args = (),
    seed: Optional[int] = None,
) -> Iterator[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
]:
    total_rewards = np.zeros((nb_assets, nb_samples))

    lifetimes_gen = lifetimes_generator(
        model,
        nb_samples,
        nb_assets,
        end_time,
        model_args=model_args,
        model1=model1,
        model1_args=model1_args,
        seed=seed,
    )

    def sample_routine(reward_func, args):
        nonlocal total_rewards  # modify these variables
        lifetimes, event_times, events, still_valid = next(lifetimes_gen)
        rewards = reward_func(lifetimes, *args)
        discountings = exponential_discounting.factor(event_times, discounting_rate)
        total_rewards += rewards * discountings
        return lifetimes, event_times, total_rewards, events, still_valid

    if reward1 is None:
        reward1 = reward
        reward1_args = reward_args

    if model1:
        try:
            yield sample_routine(reward1, reward1_args)
        except StopIteration:
            return

    while True:
        try:
            yield sample_routine(reward, reward_args)
        except StopIteration:
            break
    return


def nhpp_generator(
    model: LifetimeModel[*ModelArgs],
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: ModelArgs = (),
    seed: Optional[int] = None,
) -> Iterator[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ]
]:
    hpp_timeline = np.zeros((nb_assets, nb_samples))
    previous_timeline = np.zeros((nb_assets, nb_samples))
    still_valid = np.ones_like(hpp_timeline, dtype=np.bool_)
    exponential_dist = Exponential(1.0)

    def sample_routine(target_model, args):
        nonlocal hpp_timeline, previous_timeline, still_valid, seed  # modify these variables
        lifetimes = model_rvs(
            exponential_dist, size, model_args=args, seed=seed
        ).reshape((nb_assets, nb_samples))
        hpp_timeline += lifetimes
        ages = target_model.ichf(hpp_timeline, *args)
        durations = ages - previous_timeline
        previous_timeline = ages
        still_valid = ages < end_time
        if seed is not None:
            seed += 1
        return durations, ages, still_valid

    size = rvs_size(nb_samples, nb_assets, model_args)
    while True:
        gen_data = sample_routine(model, model_args)
        if np.any(gen_data[-1]) > 0:
            yield gen_data
        else:
            break
    return


def nhpp_policy_generator(
    model: LifetimeModel[*ModelArgs],
    ar: NDArray[np.float64],  # ages of replacement
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: ModelArgs = (),
    seed: Optional[int] = None,
) -> Iterator[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ]
]:
    hpp_timeline = np.zeros((nb_assets, nb_samples))
    previous_event_times = np.zeros((nb_assets, nb_samples))
    timeline = np.zeros((nb_assets, nb_samples))
    nb_repairs = np.zeros((nb_assets, nb_samples), dtype=np.int64)
    still_valid = np.ones_like(hpp_timeline, dtype=np.bool_)
    exponential_dist = Exponential(1.0)
    ar = np.broadcast_to(ar.reshape(-1, 1), (nb_assets, nb_samples))
    # milestones = ar.copy()  # ar milestones

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

        still_valid = timeline < end_time

        if seed is not None:
            seed += 1
        return timeline, durations, nb_repairs, still_valid

    size = rvs_size(nb_samples, nb_assets, model_args)
    while True:
        gen_data = sample_routine(model, model_args)
        if np.any(gen_data[-1]) > 0:
            yield gen_data
        else:
            break
    return
