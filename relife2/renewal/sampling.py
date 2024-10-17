from dataclasses import dataclass, field, fields
from functools import singledispatchmethod
from typing import Optional, TypeVarTuple, Iterator

import numpy as np
from numpy.typing import NDArray

from relife2.model import LifetimeModel
from relife2.renewal.discountings import Discounting
from relife2.renewal.policy import (
    Policy,
    OneCycleRunToFailure,
    OneCycleAgeReplacementPolicy,
    RunToFailure,
)
from relife2.renewal.process import RenewalProcess, RenewalRewardProcess
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
    events = np.zeros((nb_assets, nb_samples))
    still_valid = event_times < end_time

    def sample_routine(target_model, args):
        nonlocal event_times, events, still_valid  # modify these variables
        lifetimes = model_rvs(target_model, size, args=args).reshape(
            (nb_assets, nb_samples)
        )
        event_times += lifetimes
        events += 1
        still_valid = event_times < end_time
        return lifetimes, event_times, events, still_valid

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
        lifetimes, event_times, events, still_valid = next(lifetimes_gen)
        rewards = target_reward(lifetimes, *args)
        discountings = discounting.factor(event_times, *discount_args)
        total_rewards += rewards * discountings
        return lifetimes, event_times, total_rewards, events, still_valid

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
    samples: NDArray[np.int64] = field(repr=False)
    assets: NDArray[np.int64] = field(repr=False)
    events: NDArray[np.int64] = field(repr=False)
    event_times: NDArray[np.float64] = field(repr=False)

    nb_samples: int = field(init=False)
    nb_assets: int = field(init=False)
    samples_index: NDArray[np.int64] = field(init=False, repr=False)
    assets_index: NDArray[np.int64] = field(init=False, repr=False)

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
            (self.data.events, self.data.assets, self.data.samples)
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


class Sampler:
    """
    benefits of decoupling sample from other class:
    1. avoid polluting RenewalProcess with sample making inheritance
    between RenewalProcess and RenewalRewardProcess wrong (LSP) : signature overridden because of reward_args, etc.
    2. avoid having sample method that returns different dataclass and possibly non "to_lifetime_data" compatible
    => all sample returns CountData type that can be converted to lifetime data (easier to understand)
    3. meaningly speaking, one may not expect RenewalProcess or Policy object having a "sample" in its interface
    4. (3 bis) if RenewalProcess and Policy have "sample", why not LifetimeModel ? => Sampler accept LifetimeModel too
    """

    def __init__(
        self,
        end_time: float,
        nb_sample: int,
        *,
        random_state: Optional[int] = None,
    ):
        self.end_time = end_time
        self.nb_sample = nb_sample
        self.random_state = random_state
        self.sample_result, self.events, self.args, self.a0 = None, None, None, None

    def run(
        self, target_obj: LifetimeModel | RenewalProcess | Policy, **kwargs
    ) -> CountData:
        """
        Args:
            target_obj (): obj used to sample
            **kwargs (): arguments needed by policy or renewal process
        Returns:
        """
        try:
            self.sample_result, self.events, self.args, self.a0 = self.sample(
                target_obj, **kwargs
            )
        except NotImplementedError:
            raise NotImplementedError(
                f"Cannot run sampler on {type(target_obj)} object"
            )
        return self.sample_result

    @singledispatchmethod
    def sample(
        self,
        target_obj,
        **kwargs,
    ) -> tuple[
        CountData,
        Optional[NDArray[np.float64]],
        Optional[NDArray[np.float64]],
        Optional[NDArray[np.float64]],
    ]:
        """
        Args:
            target_obj (): obj used to sample
            **kwargs (): arguments needed by policy or renewal process
        Returns:
        """
        raise NotImplementedError

    @sample.register
    def _(self, target_obj: LifetimeModel, **kwargs):
        pass

    @sample.register
    def _(self, target_obj: RenewalProcess, **kwargs):
        pass

    @sample.register
    def _(self, target_obj: RenewalRewardProcess, **kwargs):
        pass

    @sample.register
    def _(self, target_obj: OneCycleRunToFailure, **kwargs):
        pass

    @sample.register
    def _(self, target_obj: OneCycleAgeReplacementPolicy, **kwargs):
        pass

    @sample.register
    def _(self, target_obj: RunToFailure, **kwargs):
        pass

    def to_lifetime_data(self):
        pass

        # if tf is None or tf > self.T:
        #     tf = self.T
        # if t0 >= tf:
        #     raise ValueError("`t0` must be strictly lesser than `tf`")
        #
        # # Filtering sample and sorting by times
        # s = self.samples == sample if sample is not None else Ellipsis
        # order = np.argsort(self.times[s])
        # indices = self.indices[s][order]
        # samples = self.samples[s][order]
        # uindices = np.ravel_multi_index(
        #     (indices, samples), (self.n_indices, self.n_samples)
        # )
        # times = self.times[s][order]
        # durations = self.durations[s][order] + self.a0[s][order]
        # events = self.events[s][order]
        #
        # # Indices of interest
        # ind0 = (times > t0) & (
        #     times <= tf
        # )  # Indices of replacement occuring inside the obervation window
        # ind1 = (
        #     times > tf
        # )  # Indices of replacement occuring after the observation window which include right censoring
        #
        # # Replacements occuring inside the observation window
        # time0 = durations[ind0]
        # event0 = events[ind0]
        # entry0 = np.zeros(time0.size)
        # _, LT = np.unique(
        #     uindices[ind0], return_index=True
        # )  # get the indices of the first replacements ocurring in the observation window
        # b0 = (
        #     times[ind0][LT] - durations[ind0][LT]
        # )  # time at birth for the firt replacements
        # entry0[LT] = np.where(b0 >= t0, 0, t0 - b0)
        # args0 = args_take(indices[ind0], *self.args)
        #
        # # Right censoring
        # _, RC = np.unique(uindices[ind1], return_index=True)
        # bf = (
        #     times[ind1][RC] - durations[ind1][RC]
        # )  # time at birth for the right censored
        # b1 = bf[
        #     bf < tf
        # ]  # ensure that time of birth for the right censored is not equal to tf.
        # time1 = tf - b1
        # event1 = np.zeros(b1.size)
        # entry1 = np.where(b1 >= t0, 0, t0 - b1)
        # args1 = args_take(indices[ind1][RC][bf < tf], *self.args)
        #
        # # Concatenate
        # time = np.concatenate((time0, time1))
        # event = np.concatenate((event0, event1))
        # entry = np.concatenate((entry0, entry1))
        # args = tuple(
        #     np.concatenate((arg0, arg1), axis=0) for arg0, arg1 in zip(args0, args1)
        # )
        # return LifetimeData(time, event, entry, args)


#     lifetimes = np.array([], dtype=np.float64)
#     failure_times = np.array([], dtype=np.float64)
#     samples = np.array([], dtype=np.int64)
#     assets = np.array([], dtype=np.int64)
#
#     for step, (_lifetimes, _failure_times, still_valid) in enumerate(
#         lifetimes_generator(
#             model,
#             nb_samples,
#             nb_assets,
#             end_time,
#             model_args=model_args,
#             delayed_model=delayed_model,
#             delayed_model_args=delayed_model_args,
#         )
#     ):
#         lifetimes = np.concatenate((lifetimes, _lifetimes[still_valid].reshape(-1)))
#         failure_times = np.concatenate(
#             (failure_times, _failure_times[still_valid].reshape(-1))
#         )
#         _assets, _samples = np.where(still_valid)
#         samples = np.concatenate((samples, _samples))
#         assets = np.concatenate((assets, _assets))
#         if nb_step:
#             if step + 1 == nb_step:
#                 break
#
#     sorted_index = np.lexsort((assets, samples))
#     lifetimes = lifetimes[sorted_index]
#     failure_times = failure_times[sorted_index]
#     samples = samples[sorted_index]
#     assets = assets[sorted_index]
#


#
#     if nb_step is None and end_time is None:
#         raise ValueError("nb_step or end_time must be given")
#     elif nb_step is not None and end_time:
#         raise ValueError("can't have nb_step and end_time given together")
#     if nb_step:
#         end_time = np.inf
#
#     failure_times = np.array([], dtype=np.float64)
#     lifetimes = np.array([], dtype=np.float64)
#     total_rewards = np.array([], dtype=np.float64)
#     samples = np.array([], dtype=np.int64)
#     assets = np.array([], dtype=np.int64)
#
#     for step, (
#         _lifetimes,
#         _failure_times,
#         _total_rewards,
#         still_valid,
#     ) in enumerate(
#         lifetimes_rewards_generator(
#             model,
#             reward,
#             discounting,
#             nb_samples,
#             nb_assets,
#             end_time,
#             model_args=model_args,
#             reward_args=reward_args,
#             discount_args=discount_args,
#             delayed_model=delayed_model,
#             delayed_model_args=delayed_model_args,
#             delayed_reward=delayed_reward,
#             delayed_reward_args=delayed_reward_args,
#         )
#     ):
#         lifetimes = np.concatenate((lifetimes, _lifetimes[still_valid].reshape(-1)))
#         failure_times = np.concatenate(
#             (failure_times, _failure_times[still_valid].reshape(-1))
#         )
#         total_rewards = np.concatenate(
#             (total_rewards, _total_rewards[still_valid].reshape(-1))
#         )
#         _assets, _samples = np.where(still_valid)
#         samples = np.concatenate((samples, _samples))
#         assets = np.concatenate((assets, _assets))
#         if nb_step:
#             if step + 1 == nb_step:
#                 break
#
#     sorted_index = np.lexsort((assets, samples))
#     lifetimes = lifetimes[sorted_index]
#     failure_times = failure_times[sorted_index]
#     total_rewards = total_rewards[sorted_index]
#     samples = samples[sorted_index]
#     assets = assets[sorted_index]
#
#     super().__init__(
#         samples,
#         assets,
#         lifetimes,
#         failure_times,
#         total_rewards,
#     )
