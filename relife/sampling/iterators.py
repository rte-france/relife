"""
timeline : calendar time of events
durations : recorded durations between events (or from t0 to first event and last event to tf)
ages : assets ages at each event
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from relife.models import (
    AgeReplacementModel,
    Exponential,
    LeftTruncatedModel,
)
from relife.rewards import Discounting, RewardsFunc
from relife.types import Args


class SampleIterator(Iterator, ABC):

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

        self.size = size
        self.tf = tf
        self.t0 = t0
        self.seed = seed

        self.timeline = None  # exposed attribute (set/get)

        # hidden attributes, control set/get interface
        self._model = None
        self._model_args = None
        self._model_type = None
        self._nb_assets = nb_assets

        self._start = None
        self._stop = None
        self._keep_last = keep_last

        self._output_dict = {}

    @property
    def nb_samples(self):
        # alias name for size
        return self.size

    @property
    def stop(self):
        """stop condition is based on a counter to keep track of last elements before tf (censoring)"""
        return np.all(self._stop > 0)

    @property
    def _just_crossed_t0(self):
        return self._start == 1

    @property
    def _just_crossed_tf(self):
        return self._stop == 1

    def output_as_dict_of_1d(self, **kwvalues):
        if self._keep_last:
            selection = np.logical_and(self._start >= 1, self._stop <= 1)
        else:
            selection = np.logical_and(self._start >= 1, self._stop < 1)

        assets_ids, samples_ids = np.where(selection)
        self._output_dict.update(
            samples_ids=samples_ids,
            assets_ids=assets_ids,
            timeline=self.timeline[selection],
        )
        self._output_dict.update(
            {k: v[selection] for k, v in kwvalues.items() if v is not None}
        )
        return self._output_dict

    @abstractmethod
    def __next__(self) -> tuple[NDArray[np.float64], ...]:
        pass


def get_nb_assets(args_tuple: tuple[Args, ...]) -> int:
    def as_2d():
        for x in args_tuple:
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)
            if len(x.shape) > 2:
                raise ValueError
            yield np.atleast_2d(x)

    return max(map(lambda x: x.shape[0], as_2d()), default=1)


class LifetimeIterator(SampleIterator):
    """
    returns time, event_indicators, entries in 2D  - shape : (nb_assets, nb_samples)
    censoring and truncations only based on t0 and tf values

    selection is done in iterable

    note that model_args is not constructed yet, in sample.py
    """

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
        self._rewards = None
        self._discounting = None
        self._a0 = None
        self._ar = None

    def set_rewards(self, rewards: RewardsFunc):
        self._rewards = rewards

    def set_discounting(self, discounting: Discounting):
        self._discounting = discounting

    def set_sampler(self, model, model_args):

        if self._model is None:
            # self._nb_assets = get_nb_assets(model_args)
            self.timeline = np.zeros((self._nb_assets, self.size))
            # counting arrays to catch values crossing t0 and tf bounds
            self._stop = np.zeros((self._nb_assets, self.size), dtype=np.int64)
            self._start = np.zeros((self._nb_assets, self.size), dtype=np.int64)

        # else:
        #     nb_assets = get_nb_assets(model_args)
        #     if nb_assets != self._nb_assets:
        #         raise ValueError("Can't change nb assets")

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

        # if self._model_type is not None:
        #     if type(model) != self._model_type:
        #         raise ValueError("Can't change model type")
        self._model = model
        self._model_type = type(model)
        self._model_args = model_args
        self._a0 = a0
        self._ar = ar

    def compute_rewards(
        self, timeline: NDArray[np.float64], durations: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        rewards = np.zeros_like(durations)
        if self._rewards and self._discounting:
            rewards = self._rewards(durations) * self._discounting.factor(timeline)
        if self._rewards and not self._discounting:
            rewards = self._rewards(durations)
        return rewards

    def _sample_routine(self) -> tuple[NDArray[np.floating], ...]:

        durations = self._model.rvs(
            *self._model_args,
            size=self.nb_samples,
            seed=self.seed,
        ).reshape((-1, self.nb_samples))
        if durations.shape != (self._nb_assets, self.nb_samples):
            # sometimes, model1 has n assets but not model
            durations = np.tile(durations, (self._nb_assets, 1))

        # create events_indicators and entries
        event_indicators = np.ones_like(self.timeline, dtype=np.bool_)
        entries = np.zeros_like(self.timeline)

        # ar right censorings
        if self._ar is not None:
            is_replaced = durations == self._ar
            event_indicators[is_replaced] = False

        # a0 left truncations
        if self._a0 is not None:
            entries = np.maximum(entries, self._a0)

        # update timeline
        self.timeline += durations

        # update start and stop counter
        self._start[self.timeline > self.t0] += 1
        self._stop[self.timeline > self.tf] += 1

        # tf right censorings
        durations = np.where(
            self._just_crossed_tf, durations - (self.timeline - self.tf), durations
        )
        self.timeline[self._just_crossed_tf] = self.tf
        event_indicators[self._just_crossed_tf] = False

        # t0 left truncations
        entries = np.where(
            self._just_crossed_t0, self.t0 - (self.timeline - durations), entries
        )
        durations = np.where(self._just_crossed_t0, durations - entries, durations)

        # update seed to avoid having the same rvs result
        if self.seed is not None:
            self.seed += 1

        return (
            durations,
            event_indicators,
            entries,
        )

    def __next__(self) -> tuple[NDArray[np.floating], ...]:
        if self._model is None:
            raise ValueError("Set sampler first")
        while not self.stop:  # recompute stop condition automatically
            durations, event_indicators, entries = self._sample_routine()
            rewards = self.compute_rewards(self.timeline, durations)

            return self.output_as_dict_of_1d(
                durations=durations,
                event_indicators=event_indicators,
                entries=entries,
                rewards=rewards,
            )

        raise StopIteration


class NonHomogeneousPoissonIterator(SampleIterator):
    """
    timeline : calendar time of events
    durations : recorded durations between events, including ar (or from t0 to first event and last event to tf)
    ages : assets ages at each event
    """

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

        self._rewards = None
        self._discounting = None

        self._hpp_timeline = None  # exposed attribute (set/get)
        self._failure_times = None
        self._ages = None
        # self._assets_ids = None
        self._ar = None
        self.is_new_asset = None
        self.entries = None
        self.renewals_ids = None
        self._exponential_dist = Exponential(1.0)

    def set_rewards(self, rewards: RewardsFunc):
        self._rewards = rewards

    def set_discounting(self, discounting: Discounting):
        self._discounting = discounting

    def compute_rewards(
        self, timeline: NDArray[np.float64], durations: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        rewards = np.zeros_like(durations)
        if self._rewards and self._discounting:
            rewards = self._rewards(durations) * self._discounting.factor(timeline)
        if self._rewards and not self._discounting:
            rewards = self._rewards(durations)
        return rewards

    def set_sampler(self, model, model_args, ar: Optional[NDArray[np.float64]] = None):

        if self._model is None:
            # self._nb_assets = get_nb_assets(model_args)
            self.timeline = np.zeros((self._nb_assets, self.size))
            # counting arrays to catch values crossing t0 and tf bounds
            self._stop = np.zeros((self._nb_assets, self.size), dtype=np.int64)
            self._start = np.zeros((self._nb_assets, self.size), dtype=np.int64)

            self._hpp_timeline = np.zeros((self._nb_assets, self.size))
            self._failure_times = np.zeros((self._nb_assets, self.size))
            self._ages = np.zeros((self._nb_assets, self.size))
            self.entries = np.zeros((self._nb_assets, self.size))
            self.is_new_asset = np.zeros(
                (self._nb_assets, self.size), dtype=np.bool_
            )  # flag to indicate replacement
            # self._assets_ids = np.arange(
            #     self._nb_assets * self.size, dtype=np.int64
            # ).reshape(self._nb_assets, self.size)
            self.renewals_ids = np.zeros((self._nb_assets, self.size), dtype=np.int64)

        self._model = model
        self._model_type = type(model)
        self._model_args = model_args
        self._ar = (
            ar if ar is not None else (np.ones(self._nb_assets) * np.inf).reshape(-1, 1)
        )

    def _sample_routine(self) -> tuple[NDArray[np.floating], ...]:
        """
        return ages : np.nan or float if value is not a0 or af

        #  keep is_new_asset (is_replaced in memory), return age at ar and reset ages, hpp_timeline, failure_times
        #  at beginning of sample_routine

        # if replaced
        # return bool indicators flagging replacement occured (True)
        # return censored age (ar)
        # return bool indicator flagging event indicators to False
        # entries is always previous ages except at t0

        # if not replaced
        # return bool indicators flagging replacement occured (False)
        # return age of repair
        # return bool indicator flagging event indicators to True
        # entries is always previous ages except at t0

        """

        # reset those who are replaced
        self._ages[self.is_new_asset] = 0.0  # asset is replaced (0 aged asset)
        # self._assets_ids[self.is_new_asset] += 1 # asset is replaced (new asset id)
        self._hpp_timeline[self.is_new_asset] = 0.0  # reset timeline
        self._failure_times[self.is_new_asset] = 0.0
        self.entries[self.is_new_asset] = 0.0
        self.renewals_ids[self.is_new_asset] += 1
        self.is_new_asset.fill(False)  # reset to False

        # generate new values
        self._hpp_timeline += self._exponential_dist.rvs(
            size=self.nb_samples * self._nb_assets, seed=self.seed
        ).reshape((self._nb_assets, self.nb_samples))

        failure_times = self._model.ichf(self._hpp_timeline, *self._model_args)
        durations = failure_times - self._failure_times  # t_i+1 - t_i
        self._failure_times = failure_times.copy()  # update t_i <- t_i+1
        self.timeline += durations
        self._ages += durations

        # create array of events_indicators
        events_indicators = np.ones_like(self._ages, np.bool_)

        # ar update (before because it changes timeline, thus start and stop conditions)
        self.timeline = np.where(
            self._ages >= self._ar,
            self.timeline - (self._ages - np.ones_like(self.timeline) * self._ar),
            self.timeline,
        )  # substract time after ar
        self._ages = np.where(
            self._ages >= self._ar, np.ones_like(self._ages) * self._ar, self._ages
        )  # set ages to ar
        self.is_new_asset[
            np.logical_and(self._ages >= self._ar, ~self._just_crossed_tf)
        ] = True
        events_indicators[
            np.logical_and(self._ages >= self._ar, ~self._just_crossed_tf)
        ] = False

        # update stop conditions
        self._start[self.timeline > self.t0] += 1
        self._stop[self.timeline > self.tf] += 1

        # t0 entries update
        self.entries = np.where(
            self._just_crossed_t0, self._ages - (self.timeline - self.t0), self.entries
        )

        # tf update censoring update
        self._ages = np.where(
            self._just_crossed_tf, self._ages - (self.timeline - self.tf), self._ages
        )
        self.timeline[self._just_crossed_tf] = self.tf
        events_indicators[self._just_crossed_tf] = False

        # returned entries
        entries = self.entries.copy()
        # update for next iteration (keep previous ages)
        self.entries = np.where(
            events_indicators, self._ages, self.entries
        )  # keep previous ages as entry for next iteration

        # update seed to avoid having the same rvs result
        if self.seed is not None:
            self.seed += 1

        return (
            self._ages,
            self.renewals_ids,
            entries,
            events_indicators,
        )

    def __next__(self) -> tuple[NDArray[np.floating], ...]:
        if self._model is None:
            raise ValueError("Set sampler first")
        while not self.stop:  # recompute stop condition automatically
            (
                ages,
                renewals_ids,
                entries,
                events_indicators,
            ) = self._sample_routine()
            rewards = self.compute_rewards(self.timeline, ages)
            # samples_ids = np.tile(np.arange(self.size).astype(np.str_), (self._nb_assets, 1))
            # prefix = np.char.add(samples_ids, np.full_like(samples_ids, "-", dtype=np.str_))
            # assets_ids = np.char.add(prefix, assets_ids.astype(np.str_))

            return self.output_as_dict_of_1d(
                ages=ages,
                renewals_ids=renewals_ids,
                entries=entries,
                events_indicators=events_indicators,
                rewards=rewards,
            )
        raise StopIteration
