from dataclasses import dataclass, field, fields
from typing import Optional, TypeVarTuple, Iterator

import numpy as np
from numpy.typing import NDArray

from relife2.model import LifetimeModel
from relife2.renewal.discountings import Discounting
from relife2.renewal.rewards import Reward

ModelArgs = TypeVarTuple("ModelArgs")
DelayedModelArgs = TypeVarTuple("DelayedModelArgs")
RewardArgs = TypeVarTuple("RewardArgs")
DelayedRewardArgs = TypeVarTuple("DelayedRewardArgs")
DiscountingArgs = TypeVarTuple("DiscountingArgs")


def model_rvs(
    model: LifetimeModel[*ModelArgs],
    size: int,
    args: tuple[*ModelArgs] | tuple[()] = (),
):
    return model.rvs(*args, size=size)


def rvs_size(
    nb_samples: int,
    nb_assets: int,
    model_args: tuple[NDArray[np.float64], ...] | tuple[()] = (),
):
    if bool(model_args) and model_args[0].ndim == 2:
        size = nb_samples  # rvs size
    else:
        size = nb_samples * nb_assets
    return size


def compute_rewards(
    reward: Reward[*RewardArgs],
    lifetimes: NDArray[np.float64],
    args: tuple[*RewardArgs] | tuple[()] = (),
):
    return reward(lifetimes, *args)


def lifetimes_generator(
    model: LifetimeModel[*ModelArgs],
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: tuple[*ModelArgs] | tuple[()] = (),
    delayed_model: Optional[LifetimeModel[*DelayedModelArgs]] = None,
    delayed_model_args: tuple[*DelayedRewardArgs] | tuple[()] = (),
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
        still_valid = event_times < end_time
        return lifetimes, event_times, order, still_valid

    if delayed_model:
        size = rvs_size(nb_samples, nb_assets, delayed_model_args)
        gen_data = sample_routine(delayed_model, delayed_model_args)
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
    discounting: Discounting[*DiscountingArgs],
    nb_samples: int,
    nb_assets: int,
    end_time: float,
    *,
    model_args: tuple[*ModelArgs] | tuple[()] = (),
    reward_args: tuple[*RewardArgs] | tuple[()] = (),
    discount_args: tuple[*DiscountingArgs] | tuple[()] = (),
    delayed_model: Optional[LifetimeModel[*DelayedModelArgs]] = None,
    delayed_model_args: tuple[*DelayedModelArgs] | tuple[()] = (),
    delayed_reward: Optional[Reward[*DelayedRewardArgs]] = None,
    delayed_reward_args: tuple[*DelayedRewardArgs] | tuple[()] = (),
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
        delayed_model=delayed_model,
        delayed_model_args=delayed_model_args,
    )

    def sample_routine(target_reward, args):
        nonlocal total_rewards  # modify these variables
        lifetimes, event_times, order, still_valid = next(lifetimes_gen)
        rewards = target_reward(lifetimes, *args)
        discountings = discounting.factor(event_times, *discount_args)
        total_rewards += rewards * discountings
        return lifetimes, event_times, total_rewards, order, still_valid

    if delayed_reward is None:
        delayed_reward = reward
        delayed_reward_args = reward_args

    if delayed_model:
        try:
            yield sample_routine(delayed_reward, delayed_reward_args)
        except StopIteration:
            return

    while True:
        try:
            yield sample_routine(reward, reward_args)
        except StopIteration:
            break
    return


@dataclass
class CountData:
    samples: NDArray[np.int64] = field(repr=False)  # samples index
    assets: NDArray[np.int64] = field(repr=False)  # assets index
    order: NDArray[np.int64] = field(repr=False)  # order index
    event_times: NDArray[np.float64] = field(repr=False)

    nb_samples: int = field(init=False)
    nb_assets: int = field(init=False)
    samples_index: NDArray[np.int64] = field(
        init=False, repr=False
    )  # unique samples index
    assets_index: NDArray[np.int64] = field(
        init=False, repr=False
    )  # unique assets index

    def __post_init__(self):
        fields_values = [
            getattr(self, _field.name) for _field in fields(self) if _field.init
        ]
        if not all(arr.ndim == 1 for arr in fields_values):
            raise ValueError("All array values must be 1d")
        if not len(set(arr.shape[0] for arr in fields_values)) == 1:
            raise ValueError("All array values must have the same shape")

        self.samples_index = np.unique(self.samples)
        self.assets_index = np.unique(self.assets)
        self.nb_samples = len(self.samples_index)
        self.nb_assets = len(self.assets_index)

    def __len__(self) -> int:
        return self.nb_samples * self.nb_assets

    def number_of_events(
        self, sample: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ind = self.samples == sample
        times = np.insert(np.sort(self.event_times[ind]), 0, 0)
        counts = np.arange(times.size)
        return times, counts

    def mean_number_of_events(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        times = np.insert(np.sort(self.event_times), 0, 0)
        counts = np.arange(times.size) / self.nb_samples
        return times, counts

    def iter(self):
        return CountDataIterable(self)


class CountDataIterable:
    def __init__(self, data: CountData):
        self.data = data

        sorted_index = np.lexsort(
            (self.data.order, self.data.assets, self.data.samples)
        )
        self.sorted_fields = {
            _field.name: getattr(self.data, _field.name).copy()[sorted_index]
            for _field in fields(self.data)
            if _field.init
        }

    def __len__(self) -> int:
        return self.data.nb_samples * self.data.nb_assets

    def __iter__(self) -> Iterator[tuple[int, int, dict[str, NDArray[np.float64]]]]:

        for sample in self.data.samples_index:
            sample_mask = self.sorted_fields["samples"] == sample
            for asset in self.data.assets_index:
                asset_mask = self.sorted_fields["assets"][sample_mask] == asset
                values_dict = {
                    k: v[sample_mask][asset_mask]
                    for k, v in self.sorted_fields.items()
                    if k not in ("samples", "assets", "events")
                }
                yield int(sample), int(asset), values_dict


@dataclass
class RenewalData(CountData):
    lifetimes: NDArray[np.float64] = field(repr=False)


@dataclass
class RenewalRewardData(RenewalData):
    total_rewards: NDArray[np.float64] = field(repr=False)

    def cum_total_rewards(
        self, sample: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ind = self.samples == sample
        s = np.argsort(self.event_times[ind])
        times = np.insert(self.event_times[ind][s], 0, 0)
        z = np.insert(self.total_rewards[ind][s].cumsum(), 0, 0)
        return times, z

    def mean_total_reward(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        s = np.argsort(self.event_times)
        times = np.insert(self.event_times[s], 0, 0)
        z = np.insert(self.total_rewards[s].cumsum(), 0, 0) / self.nb_samples
        return times, z
