from typing import Iterator, Optional

import numpy as np
from numpy.typing import NDArray

from relife2.fiability.addon import AgeReplacementModel
from relife2.fiability.model import LifetimeModel
from relife2.renewal.discount import Discount
from relife2.renewal.reward import Reward
from relife2.typing import DiscountArgs, Model1Args, ModelArgs, Reward1Args, RewardArgs


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
) -> Iterator[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
]:

    event_times = np.zeros((nb_assets, nb_samples))
    order = np.zeros((nb_assets, nb_samples))
    still_valid = event_times < end_time

    def sample_routine(target_model, args):
        nonlocal event_times, order, still_valid  # modify these variables
        lifetimes = model_rvs(target_model, size, args=args).reshape(
            (nb_assets, nb_samples)
        )
        event_times += lifetimes
        order += 1
        events = compute_events(lifetimes, target_model, args)
        still_valid = event_times < end_time
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
    reward: Reward[*RewardArgs],
    discounting: Discount[*DiscountArgs],
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: ModelArgs = (),
    reward_args: RewardArgs = (),
    discount_args: DiscountArgs = (),
    model1: Optional[LifetimeModel[*Model1Args]] = None,
    model1_args: Model1Args = (),
    reward1: Optional[Reward[*Reward1Args]] = None,
    reward1_args: Reward1Args = (),
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
    )

    def sample_routine(target_reward, args):
        nonlocal total_rewards  # modify these variables
        lifetimes, event_times, events, still_valid = next(lifetimes_gen)
        rewards = target_reward(lifetimes, *args)
        discountings = discounting.factor(event_times, *discount_args)
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
