"""
Below iterators are used in policies to :
1. sample and create CountData object that holds sampled data to compute empirical costs and plots
2. sample and return arrays that can be used to fit the model's policy

Those iterators needs a model and model_args. They are set manually so one can decompose each iteration
steps in case there is a model1

They hold t0 and tf. tf controls the stop iteration condition. If t0 is not zero, the iterator can return
a selection of empty arrays

They also memory the timeline array internally, it serves to compute the stop condition with t0 and tf but
it is not returned.

All returned arrays are 1d.

The iterotors allows to :
- isolate the iteration code in order to reuse it in several parts
- decompose each iteration steps with next command
"""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional
import numpy as np
from collections.abc import Iterator

from numpy.typing import NDArray

from relife.core import (
    AgeReplacementModel,
    LeftTruncatedModel,
)
from relife.models import Exponential
from relife.rewards import RewardsFunc, Discounting
from relife.types import Arg


class SampleIterator(Iterator, ABC):

    def __init__(
        self,
        size: int,
        tf: float,  # calendar end time
        t0: float = 0.0,  # calendar beginning time
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
        self._nb_assets = None

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
    def _crossed_t0(self):
        return self._start == 1

    @property
    def _crossed_tf(self):
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
        self._output_dict.update({k: v[selection] for k, v in kwvalues.items()})
        return self._output_dict

    @abstractmethod
    def __next__(self) -> tuple[NDArray[np.float64], ...]:
        pass


def get_nb_assets(args_tuple: tuple[Arg, ...]) -> int:
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
        *,
        seed: Optional[int] = None,
        keep_last: bool = True,
    ):
        super().__init__(size, tf, t0, seed=seed, keep_last=keep_last)
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
            self._nb_assets = get_nb_assets(model_args)
            self.timeline = np.zeros((self._nb_assets, self.size))
            # counting arrays to catch values crossing t0 and tf bounds
            self._stop = np.zeros((self._nb_assets, self.size), dtype=np.int64)
            self._start = np.zeros((self._nb_assets, self.size), dtype=np.int64)

        else:
            nb_assets = get_nb_assets(model_args)
            if nb_assets != self._nb_assets:
                raise ValueError("Can't change nb assets")

        ar = None
        a0 = None
        if isinstance(model, AgeReplacementModel):
            ar = model_args[0].copy()
            model_args = model_args[1:]
            model = model.baseline
            if isinstance(model, LeftTruncatedModel):
                a0 = model_args[1].copy()
                model_args = model_args[1:]
                model = model.baseline
        elif isinstance(model, LeftTruncatedModel):
            a0 = model_args[0].copy()
            model_args = model_args[1:]
            model = model.baseline
            if isinstance(model, AgeReplacementModel):
                ar = model_args[1].copy()
                model_args = model_args[1:]
                model = model.baseline

        if self._model_type is not None:
            if type(model) != self._model_type:
                raise ValueError("Can't change model type")
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
            size=self.nb_samples,
            seed=self.seed,
        ).reshape((self._nb_assets, self.nb_samples))

        if self._ar:
            durations = np.minimum(durations, self._ar)
        if self._a0 is not None:
            durations = durations + self._a0

        # update timeline
        self.timeline += durations

        # update start and stop counter
        self._start[self.timeline > self.t0] += 1
        self._stop[self.timeline > self.tf] += 1

        # censor values
        event_indicators = np.ones_like(self.timeline, dtype=np.bool_)
        durations = np.where(
            self._crossed_tf, durations - (self.timeline - self.tf), durations
        )
        self.timeline[self._crossed_tf] = self.tf
        event_indicators[self._crossed_tf] = False

        # get left truncations
        entries = np.zeros_like(self.timeline)
        entries = np.where(
            self._crossed_t0, self.t0 - (self.timeline - durations), entries
        )

        # update censorings
        if self._ar is not None:
            event_indicators = np.where(durations < self._ar, event_indicators, False)
        # update truncations
        if self._a0 is not None:
            entries = np.maximum(entries, self._a0)

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
            rewards = (
                self.compute_rewards(self.timeline, durations)
                if self._rewards
                else None
            )

            return self.output_as_dict_of_1d(
                durations=durations,
                event_indicators=event_indicators,
                entries=entries,
                rewards=rewards,
            )

        raise StopIteration


class NonHomogeneousPoissonIterator(SampleIterator):
    """
    returns a0, af, ages in 2D - shape : (nb_assets, nb_samples)

    selection is done in iterable
    """

    def __init__(
        self,
        size: int,
        tf: float,  # calendar end time
        t0: float = 0.0,  # calendar beginning time
        *,
        seed: Optional[int] = None,
    ):
        super().__init__(size, tf, t0, seed=seed)

        self._hpp_timeline = None  # exposed attribute (set/get)
        self._failure_times = None
        self._ages = None
        self._ar = None
        self._exponential_dist = Exponential(1.0)

    def set_sampler(self, model, model_args, ar: Optional[NDArray[np.float64]] = None):

        if self._model is None:
            self._nb_assets = get_nb_assets(model_args)
            self.timeline = np.zeros((self._nb_assets, self.size))
            # counting arrays to catch values crossing t0 and tf bounds
            self._stop = np.zeros((self._nb_assets, self.size), dtype=np.int64)
            self._start = np.zeros((self._nb_assets, self.size), dtype=np.int64)

            self._hpp_timeline = np.zeros((self._nb_assets, self.size))
            self._failure_times = np.zeros((self._nb_assets, self.size))
            self._ages = np.zeros((self._nb_assets, self.size))

        self._model = model
        self._model_type = type(model)
        self._model_args = model_args
        self._ar = ar

    def _sample_routine(self) -> tuple[NDArray[np.floating], ...]:
        """
        return ages : np.nan or float if value is not a0 or af
        """
        self._hpp_timeline += self._exponential_dist.rvs(
            size=self.nb_samples * self._nb_assets, seed=self.seed
        ).reshape((self._nb_assets, self.nb_samples))

        failure_times = self._model.ichf(self._hpp_timeline, *self._model_args)
        durations = failure_times - self._failure_times  # t_i+1 - t_i
        self._failure_times = failure_times.copy()  # update t_i <- t_i+1
        self.timeline += durations
        self._ages += durations

        a0 = np.full_like(self.timeline, np.nan)
        af = np.full_like(self.timeline, np.nan)

        if self._ar is not None:
            is_replaced = self._ages >= self._ar
            af = np.where(is_replaced, np.ones_like(af) * self._ar, af)
            a0 = np.where(is_replaced, np.zeros_like(af), a0)
            self.timeline = np.where(
                is_replaced,
                self.timeline - (self._ages - np.ones_like(self.timeline) * self._ar),
                self.timeline,
            )
            self._hpp_timeline[is_replaced] = 0.0
            self._failure_times[is_replaced] = 0.0
            self._ages[is_replaced] = 0.0

        # update start and stop counter
        self._start[self.timeline > self.t0] += 1
        self._stop[self.timeline > self.tf] += 1

        # update a0 values
        a0[self._crossed_t0] = self.t0

        # af values
        af[self._crossed_tf] = self.tf
        durations[self._crossed_tf] = np.nan  # no reparations
        self.timeline[self._crossed_tf] = self.tf

        # update seed to avoid having the same rvs result
        if self.seed is not None:
            self.seed += 1

        return durations, a0, af

    def __next__(self) -> tuple[NDArray[np.floating], ...]:
        if self._model is None:
            raise ValueError("Set sampler first")
        while not self.stop:  # recompute stop condition automatically
            durations, a0, af = self._sample_routine()
            return self.output_as_dict_of_1d(durations=durations, a0=a0, af=af)
        raise StopIteration
