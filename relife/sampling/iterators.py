from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from relife.core import ParametricModel, ParametricLifetimeModel
from relife.models import (
    AgeReplacementModel,
    Exponential,
    LeftTruncatedModel,
)
from relife.rewards import Discounting, RewardsFunc
from relife.types import Args, NDArrayOfAny


class SampleIterator(Iterator, ABC):

    model: Optional[ParametricModel]
    model_args: Optional[tuple[Args, ...]]
    start_counter: Optional[NDArray[np.int64]]
    end_counter: Optional[NDArray[np.int64]]

    def __init__(
        self,
        size: int,  # nb samples
        tf: float,  # calendar end time
        t0: float = 0.0,  # calendar beginning time
        nb_assets: int = 1,
        *,
        seed: Optional[int] = None,
        keep_last: bool = True,
    ):

        self.size = size  # nb_samples
        self.tf = tf
        self.t0 = t0
        self.seed = seed

        self.timeline = None  # exposed attribute (set/get)

        # hidden attributes, control set/get interface
        self.model = None
        self.model_args = None
        self.nb_assets = nb_assets

        self.start_counter = None
        self.stop_counter = None
        self.keep_last = keep_last

    @property
    def stop(self) -> Optional[bool]:
        if self.stop_counter is not None:
            return np.all(self.stop_counter > 0)

    @property
    def just_crossed_t0(self) -> Optional[NDArray[np.bool_]]:
        if self.start_counter is not None:
            return self.start_counter == 1

    @property
    def just_crossed_tf(self) -> Optional[NDArray[np.bool_]]:
        if self.stop_counter is not None:
            return self.stop_counter == 1

    def select_1d(self, **kwvalues: NDArrayOfAny) -> dict[str, NDArrayOfAny]:
        output_dict = {}
        if self.keep_last:
            selection = np.logical_and(self.start_counter >= 1, self.stop_counter <= 1)
        else:
            selection = np.logical_and(self.start_counter >= 1, self.stop_counter < 1)

        assets_ids, samples_ids = np.where(selection)
        output_dict.update(
            samples_ids=samples_ids,
            assets_ids=assets_ids,
            timeline=self.timeline[selection],
        )
        output_dict.update(
            {k: v[selection] for k, v in kwvalues.items() if v is not None}
        )
        return output_dict

    @abstractmethod
    def step(self) -> dict[str, NDArrayOfAny]:
        pass

    def __next__(self) -> dict[str, NDArrayOfAny]:
        if self.model is None:
            raise ValueError("Set sampler first")
        while not self.stop:
            return self.step()
        raise StopIteration


class LifetimeIterator(SampleIterator):

    rewards: Optional[RewardsFunc]
    discounting: Optional[Discounting]
    a0: Optional[NDArray[np.float64]]
    ar: Optional[NDArray[np.float64]]

    def __init__(
        self,
        size: int,
        tf: float,  # calendar end time
        t0: float = 0.0,  # calendar start time
        nb_assets: int = 1,
        *,
        seed: Optional[int] = None,
        keep_last: bool = True,
    ):
        super().__init__(
            size, tf, t0, nb_assets=nb_assets, seed=seed, keep_last=keep_last
        )
        self.rewards = None
        self.discounting = None
        self.a0 = None
        self.ar = None

    def set_sampler(
        self, model: ParametricLifetimeModel, model_args: tuple[Args, ...] = ()
    ) -> None:

        if self.model is None:
            self.timeline = np.zeros((self.nb_assets, self.size))
            self.stop_counter = np.zeros((self.nb_assets, self.size), dtype=np.int64)
            self.start_counter = np.zeros((self.nb_assets, self.size), dtype=np.int64)

        ar = None
        a0 = None
        if isinstance(model, AgeReplacementModel):
            ar = model_args[0].copy()
            if isinstance(model, LeftTruncatedModel):
                a0 = model_args[1].copy()
        elif isinstance(model, LeftTruncatedModel):
            a0 = model_args[0].copy()
            if isinstance(model, AgeReplacementModel):
                ar = model_args[1].copy()

        self.model = model
        self.model_args = model_args
        self.a0 = a0
        self.ar = ar

    def compute_rewards(
        self, timeline: NDArray[np.float64], durations: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        rewards = np.zeros_like(durations)
        if self.rewards and self.discounting:
            rewards = self.rewards(durations) * self.discounting.factor(timeline)
        if self.rewards and not self.discounting:
            rewards = self.rewards(durations)
        return rewards

    def step(self) -> dict[str, NDArrayOfAny]:

        durations = self.model.rvs(
            *self.model_args,
            size=self.size,
            seed=self.seed,
        ).reshape((-1, self.size))
        if durations.shape != (self.nb_assets, self.size):
            # sometimes, model1 has n assets but not model
            durations = np.tile(durations, (self.nb_assets, 1))

        # create events_indicators and entries
        event_indicators = np.ones_like(self.timeline, dtype=np.bool_)
        entries = np.zeros_like(self.timeline)

        # ar right censorings
        if self.ar is not None:
            is_replaced = durations == self.ar
            event_indicators[is_replaced] = False

        # a0 left truncations
        if self.a0 is not None:
            entries = np.maximum(entries, self.a0)

        # update timeline
        self.timeline += durations

        # update start and stop counter
        self.start_counter[self.timeline > self.t0] += 1
        self.stop_counter[self.timeline > self.tf] += 1

        # tf right censorings
        durations = np.where(
            self.just_crossed_tf, durations - (self.timeline - self.tf), durations
        )
        self.timeline[self.just_crossed_tf] = self.tf
        event_indicators[self.just_crossed_tf] = False

        # t0 left truncations
        entries = np.where(
            self.just_crossed_t0, self.t0 - (self.timeline - durations), entries
        )
        durations = np.where(self.just_crossed_t0, durations - entries, durations)

        # update seed to avoid having the same rvs result
        if self.seed is not None:
            self.seed += 1

        rewards = self.compute_rewards(self.timeline, durations)
        return self.select_1d(
            durations=durations,
            event_indicators=event_indicators,
            entries=entries,
            rewards=rewards,
        )


class NonHomogeneousPoissonIterator(SampleIterator):

    rewards: Optional[RewardsFunc]
    discounting: Optional[Discounting]
    hpp_timeline: Optional[NDArray[np.float64]]
    failure_times: Optional[NDArray[np.float64]]
    ages: Optional[NDArray[np.float64]]
    ar: Optional[NDArray[np.float64]]
    is_new_asset: Optional[NDArray[np.bool_]]
    entries: Optional[NDArray[np.float64]]
    renewals_ids: Optional[NDArray[np.int64]]

    def __init__(
        self,
        size: int,
        tf: float,  # calendar end time
        t0: float = 0.0,  # calendar beginning time
        nb_assets: int = 1,
        *,
        seed: Optional[int] = None,
        keep_last: bool = True,
    ):
        super().__init__(
            size, tf, t0, nb_assets=nb_assets, seed=seed, keep_last=keep_last
        )

        self.rewards = None
        self.discounting = None

        self.hpp_timeline = None  # exposed attribute (set/get)
        self.failure_times = None
        self.ages = None
        # self._assets_ids = None
        self.ar = None
        self.is_new_asset = None
        self.entries = None
        self.renewals_ids = None
        self.exponential_dist = Exponential(1.0)

    def compute_rewards(
        self, timeline: NDArray[np.float64], ages: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        rewards = np.zeros_like(ages)
        if self.rewards and self.discounting:
            rewards = self.rewards(ages) * self.discounting.factor(timeline)
        if self.rewards and not self.discounting:
            rewards = self.rewards(ages)
        return rewards

    def set_sampler(
        self,
        model: ParametricLifetimeModel,
        model_args: tuple[Args, ...] = (),
        ar: Optional[NDArray[np.float64]] = None,
    ) -> None:

        if self.model is None:
            # self._nb_assets = get_nb_assets(model_args)
            self.timeline = np.zeros((self.nb_assets, self.size))
            # counting arrays to catch values crossing t0 and tf bounds
            self.stop_counter = np.zeros((self.nb_assets, self.size), dtype=np.int64)
            self.start_counter = np.zeros((self.nb_assets, self.size), dtype=np.int64)

            self.hpp_timeline = np.zeros((self.nb_assets, self.size))
            self.failure_times = np.zeros((self.nb_assets, self.size))
            self.ages = np.zeros((self.nb_assets, self.size))
            self.entries = np.zeros((self.nb_assets, self.size))
            self.is_new_asset = np.zeros((self.nb_assets, self.size), dtype=np.bool_)
            self.renewals_ids = np.zeros((self.nb_assets, self.size), dtype=np.int64)

        self.model = model
        self.model_args = model_args
        self.ar = (
            ar if ar is not None else (np.ones(self.nb_assets) * np.inf).reshape(-1, 1)
        )

    def step(self) -> dict[str, NDArrayOfAny]:

        # reset those who are replaced
        self.ages[self.is_new_asset] = 0.0  # asset is replaced (0 aged asset)
        # self._assets_ids[self.is_new_asset] += 1 # asset is replaced (new asset id)
        self.hpp_timeline[self.is_new_asset] = 0.0  # reset timeline
        self.failure_times[self.is_new_asset] = 0.0
        self.entries[self.is_new_asset] = 0.0
        self.renewals_ids[self.is_new_asset] += 1
        self.is_new_asset.fill(False)  # reset to False

        # generate new values
        self.hpp_timeline += self.exponential_dist.rvs(
            size=self.size * self.nb_assets, seed=self.seed
        ).reshape((self.nb_assets, self.size))

        failure_times = self.model.ichf(self.hpp_timeline, *self.model_args)
        durations = failure_times - self.failure_times  # t_i+1 - t_i
        self.failure_times = failure_times.copy()  # update t_i <- t_i+1
        self.timeline += durations
        self.ages += durations

        # create array of events_indicators
        events_indicators = np.ones_like(self.ages, np.bool_)

        # ar update (before because it changes timeline, thus start and stop conditions)
        self.timeline = np.where(
            self.ages >= self.ar,
            self.timeline - (self.ages - np.ones_like(self.timeline) * self.ar),
            self.timeline,
        )  # substract time after ar
        self.ages = np.where(
            self.ages >= self.ar, np.ones_like(self.ages) * self.ar, self.ages
        )  # set ages to ar
        self.is_new_asset[
            np.logical_and(self.ages >= self.ar, ~self.just_crossed_tf)
        ] = True
        events_indicators[
            np.logical_and(self.ages >= self.ar, ~self.just_crossed_tf)
        ] = False

        # update stop conditions
        self.start_counter[self.timeline > self.t0] += 1
        self.stop_counter[self.timeline > self.tf] += 1

        # t0 entries update
        self.entries = np.where(
            self.just_crossed_t0, self.ages - (self.timeline - self.t0), self.entries
        )

        # tf update censoring update
        self.ages = np.where(
            self.just_crossed_tf, self.ages - (self.timeline - self.tf), self.ages
        )
        self.timeline[self.just_crossed_tf] = self.tf
        events_indicators[self.just_crossed_tf] = False

        # returned entries
        entries = self.entries.copy()
        # update for next iteration (keep previous ages)
        self.entries = np.where(
            events_indicators, self.ages, self.entries
        )  # keep previous ages as entry for next iteration

        # update seed to avoid having the same rvs result
        if self.seed is not None:
            self.seed += 1

        rewards = self.compute_rewards(self.timeline, self.ages)

        return self.select_1d(
            ages=self.ages,
            renewals_ids=self.renewals_ids,
            entries=entries,
            events_indicators=events_indicators,
            rewards=rewards,
        )
