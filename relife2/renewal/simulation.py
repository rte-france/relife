from functools import singledispatchmethod
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from relife2.fiability.addons import AgeReplacementModel
from relife2.model import LifetimeModel
from relife2.renewal.discountings import exponential_discounting
from relife2.renewal.policy import (
    Policy,
    OneCycleAgeReplacementPolicy,
    OneCycleRunToFailure,
    RunToFailure,
)
from relife2.renewal.process import RenewalRewardProcess, RenewalProcess
from relife2.renewal.rewards import run_to_failure_cost
from relife2.renewal.sampling import (
    lifetimes_rewards_generator,
    RenewalData,
    lifetimes_generator,
    CountData,
    RenewalRewardData,
)


class Simulator:
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
        nb_samples: int,
        nb_assets: int,
        end_time: float,
        *,
        random_state: Optional[int] = None,
    ):
        self.end_time = end_time
        self.nb_samples = nb_samples
        self.nb_assets = nb_assets
        self.random_state = random_state
        # treat args and a0 directly in to_lifetime_data if possible
        # a0 only exists for LeftTruncated Model, else it is 0 (see AgeReplacementPolicy)
        # checks if model is LeftTruncated can be made there
        self.count_data, self.events = None, None

    def run(
        self, source: LifetimeModel | RenewalProcess | Policy, **kwargs: Any
    ) -> CountData:
        """
        Args:
            source (): obj used to sample
            **kwargs (): arguments needed by policy or renewal process
        Returns:
        """
        try:
            self.count_data, self.events = self.sample(source, **kwargs)
        except NotImplementedError:
            raise NotImplementedError(f"Cannot run sampler on {type(source)} object")
        return self.count_data

    @singledispatchmethod
    def sample(
        self,
        source,
        **kwargs,
    ) -> tuple[
        CountData,
        NDArray[np.float64],
    ]:
        """
        single dispatch method with covariant return type (return type is subtype of CountData)
        Args:
            source (): obj used to sample
            **kwargs (): arguments needed by policy or renewal process
        Returns:
        """
        raise NotImplementedError

    @sample.register
    def _(self, source: LifetimeModel, **kwargs):
        lifetimes = np.array([], dtype=np.float64)
        event_times = np.array([], dtype=np.float64)
        samples = np.array([], dtype=np.int64)
        assets = np.array([], dtype=np.int64)
        order = np.array([], dtype=np.int64)

        for *gen_data, still_valid in lifetimes_generator(
            source,
            self.nb_samples,
            self.nb_assets,
            end_time=self.end_time,
            model_args=tuple(kwargs.values()),
        ):
            lifetimes = np.concatenate((lifetimes, gen_data[0][still_valid]))
            event_times = np.concatenate((event_times, gen_data[1][still_valid]))
            order = np.concatenate((order, gen_data[2][still_valid]))
            _assets, _samples = np.where(still_valid)
            samples = np.concatenate((samples, _samples))
            assets = np.concatenate((assets, _assets))

        count_data = RenewalData(samples, assets, order, event_times, lifetimes)
        events = np.ones(len(count_data))
        if isinstance(source, AgeReplacementModel):
            ar = kwargs["ar"]
            events[
                count_data.lifetimes < np.take(count_data.assets, ar.reshape(-1))
            ] = 0
        return count_data, events

    @sample.register
    def _(self, source: RenewalProcess, **kwargs):
        lifetimes = np.array([], dtype=np.float64)
        event_times = np.array([], dtype=np.float64)
        samples = np.array([], dtype=np.int64)
        assets = np.array([], dtype=np.int64)
        order = np.array([], dtype=np.int64)

        for *gen_data, still_valid in lifetimes_generator(
            source.model,
            self.nb_samples,
            self.nb_assets,
            end_time=self.end_time,
            model_args=kwargs["model_args"],
            delayed_model=source.delayed_model,
            delayed_model_args=kwargs["delayed_model_args"],
        ):
            lifetimes = np.concatenate((lifetimes, gen_data[0][still_valid]))
            event_times = np.concatenate((event_times, gen_data[1][still_valid]))
            order = np.concatenate((order, gen_data[2][still_valid]))
            _assets, _samples = np.where(still_valid)
            samples = np.concatenate((samples, _samples))
            assets = np.concatenate((assets, _assets))

        count_data = RenewalData(samples, assets, order, event_times, lifetimes)
        events = np.ones(len(count_data))
        if isinstance(source.delayed_model, AgeReplacementModel):
            ar = kwargs["delayed_model_args"][0]
            events[
                count_data.lifetimes[count_data.order == 1]
                < np.take(count_data.assets[count_data.order == 1], ar.reshape(-1))
            ] = 0

        if isinstance(source.model, AgeReplacementModel):
            ar = kwargs["model_args"][0]
            events[
                count_data.lifetimes[count_data.order > 1]
                < np.take(count_data.assets[count_data.order > 1], ar.reshape(-1))
            ] = 0

        return count_data, events

    @sample.register
    def _(self, source: RenewalRewardProcess, **kwargs):
        lifetimes = np.array([], dtype=np.float64)
        event_times = np.array([], dtype=np.float64)
        total_rewards = np.array([], dtype=np.float64)
        samples = np.array([], dtype=np.int64)
        assets = np.array([], dtype=np.int64)
        order = np.array([], dtype=np.int64)

        for (
            *gen_data,
            still_valid,
        ) in lifetimes_rewards_generator(
            source.model,
            source.reward,
            exponential_discounting,
            self.nb_samples,
            self.nb_assets,
            end_time=self.end_time,
            model_args=kwargs["model_args"],
            reward_args=kwargs["reward_args"],
            discount_args=(kwargs["discounting_rate"],),
            delayed_model=source.delayed_model,
            delayed_model_args=kwargs["delayed_model_args"],
            delayed_reward=source.delayed_reward,
            delayed_reward_args=kwargs["delayed_reward_args"],
        ):
            lifetimes = np.concatenate((lifetimes, gen_data[0][still_valid]))
            event_times = np.concatenate((event_times, gen_data[1][still_valid]))
            total_rewards = np.concatenate((total_rewards, gen_data[2][still_valid]))
            order = np.concatenate((order, gen_data[3][still_valid]))
            _assets, _samples = np.where(still_valid)
            samples = np.concatenate((samples, _samples))
            assets = np.concatenate((assets, _assets))

        count_data = RenewalRewardData(
            samples, assets, order, event_times, lifetimes, total_rewards
        )
        events = np.ones(len(count_data))
        if isinstance(source.delayed_model, AgeReplacementModel):
            ar = kwargs["delayed_model_args"][0]
            events[
                count_data.lifetimes[count_data.order == 1]
                < np.take(count_data.assets[count_data.order == 1], ar.reshape(-1))
            ] = 0

        if isinstance(source.model, AgeReplacementModel):
            ar = kwargs["model_args"][0]
            events[
                count_data.lifetimes[count_data.order > 1]
                < np.take(count_data.assets[count_data.order > 1], ar.reshape(-1))
            ] = 0

    @sample.register
    def _(self, source: OneCycleRunToFailure, **kwargs):
        generator = lifetimes_rewards_generator(
            source.model,
            run_to_failure_cost,
            exponential_discounting,
            self.nb_samples,
            self.nb_assets,
            np.inf,
            model_args=source.model_args,
            reward_args=(kwargs["cf"],),
            discount_args=(kwargs["rate"],),
        )

        lifetimes, event_times, total_rewards, order, still_valid = next(generator)
        assets, samples = np.where(still_valid)
        assets.astype(np.int64)
        samples.astype(np.int64)

        count_data = RenewalRewardData(
            samples, assets, order, event_times, lifetimes, total_rewards
        )
        events = np.ones(len(count_data))
        return count_data, events

    @sample.register
    def _(self, source: OneCycleAgeReplacementPolicy, **kwargs: Any):
        pass

    @sample.register
    def _(self, source: RunToFailure, **kwargs: Any):
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
