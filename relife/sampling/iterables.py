from typing import Iterable, Optional, Callable

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife.core import LifetimeModel, AgeReplacementModel, LeftTruncatedModel
from .iterators import LifetimeIterator, NonHomogeneousPoissonIterator
from relife.types import Arg
from ..policies import NonHomogeneousPoissonAgeReplacementPolicy
from ..process import NonHomogeneousPoissonProcess


class RenewalIterable(Iterable):
    """
    Iterable returning dict of 1D arrays selected from each iterations:
        - samples_ids
        - assets_ids
        - timeline
        - durations
        - event_indicators
        - entries
        - rewards
    when RenewalIterable is used, one can pick only needed data

    reward_func and discount_func are optional but must take one array (durations) and return one array (costs)
    """

    def __init__(
        self,
        size: int,
        tf: float,
        model: LifetimeModel[*tuple[NDArray[np.float64], ...]],
        reward_func: Optional[
            Callable[[NDArray[np.float64]], NDArray[np.float64]]
        ] = None,
        discount_factor: Optional[
            Callable[[NDArray[np.float64]], NDArray[np.float64]]
        ] = None,
        model_args: tuple[Arg, ...] = (),
        model1: Optional[LifetimeModel[*tuple[Arg, ...]]] = None,
        model1_args: tuple[Arg, ...] = (),
        t0: float = 0.0,
        seed: Optional[int] = None,
        keep_last: bool = True,
    ):
        self.size = size
        self.tf = tf
        self.t0 = t0
        self.keep_last = keep_last

        # control model1/model types and extract a0, ar and ar1 values to compute truncations and censorings
        self.ar = None
        if isinstance(model, AgeReplacementModel):
            self.ar = model_args[0].copy()
            self.model_args = model_args
            if isinstance(model.baseline, LeftTruncatedModel):
                raise ValueError("LefTruncatedModel is allowed for model1 only")
        self.model = model
        self.model_args = model_args

        self.a0 = None
        self.ar1 = None
        if model1 is not None:
            if isinstance(model1, AgeReplacementModel):
                self.ar1 = model1_args[0].copy()
                if isinstance(model1.baseline, LeftTruncatedModel):
                    self.a0 = model1_args[1].copy()
            elif isinstance(model1, LeftTruncatedModel):
                self.a0 = model1_args[0].copy()
                if isinstance(model1.baseline, AgeReplacementModel):
                    self.ar1 = model1_args[1].copy()
            self.model1 = model1
            self.model1_args = model1_args
        else:
            self.model1 = model
            self.model1_args = model_args

        self.reward_func = reward_func
        self.discount_func = discount_factor
        self.seed = seed

        self.returned_dict = {
            "samples_ids": np.array([], dtype=np.int64),
            "assets_ids": np.array([], dtype=np.int64),
            "timeline": np.array([], dtype=np.float64),
            "durations": np.array([], dtype=np.float64),
            "event_indicators": np.array([], dtype=np.float64),
            "entries": np.array([], dtype=np.float64),
            "rewards": np.array([], dtype=np.float64),
        }

    def compute_rewards(self, timeline, durations):
        rewards = np.zeros_like(durations)
        if self.reward_func and self.discount_func:
            rewards = self.reward_func(durations) * self.discount_func(timeline)
        if self.reward_func and not self.discount_func:
            rewards = self.reward_func(durations)
        return rewards

    @staticmethod
    def update_truncations(entries, a0: NDArray[np.float64]):
        return np.maximum(entries, a0)

    @staticmethod
    def update_censorings(durations, event_indicators, ar: NDArray[np.float64]):
        return np.where(durations < ar, event_indicators, False)

    def update_returned_dict(
        self, timeline, durations, event_indicators, entries, ar=None, a0=None
    ):
        # select values in t0 tf observation window
        if self.keep_last:
            timeline_selection = np.logical_and(
                self.t0 <= timeline, timeline <= self.tf
            )
        else:
            timeline_selection = np.logical_and(self.t0 <= timeline, timeline < self.tf)

        # get ids
        assets_ids, samples_ids = np.where(timeline_selection)

        # compute rewards
        rewards = self.compute_rewards(timeline, durations)

        # compute event/entries based on ar/a0
        if ar is not None:
            event_indicators = RenewalIterable.update_censorings(
                durations, event_indicators, ar=ar
            )
        if a0 is not None:
            entries = RenewalIterable.update_truncations(entries, a0=a0)
        #  compute rewards

        self.returned_dict.update(
            samples_ids=samples_ids,
            assets_ids=assets_ids,
            timeline=timeline[timeline_selection],
            durations=durations[timeline_selection],
            event_indicators=event_indicators[timeline_selection],
            entries=entries[timeline_selection],
            rewards=rewards[timeline_selection],
        )

    def __iter__(self):
        # construct iterator
        iterator = LifetimeIterator(self.size, self.tf, self.t0, seed=self.seed)

        # first cycle : set model1 in iterator
        iterator.load(self.model1, self.model1_args)

        # yield first dict of results (return empty arrays if iterator stops at first iteration)
        try:
            durations, event_indicators, entries = next(
                iterator
            )  # 2d returns (nb_assets, nb_samples)

            timeline = iterator.timeline
            self.update_returned_dict(
                timeline, durations, event_indicators, entries, ar=self.ar1, a0=self.a0
            )
            yield self.returned_dict
        except StopIteration:
            yield self.returned_dict

        # next cycles : set model in iterator
        iterator.load(self.model, self.model_args)

        # yield dict of results until iterator stops
        for durations, event_indicators, entries in iterator:
            # select values in t0 tf observation window
            timeline = iterator.timeline
            self.update_returned_dict(
                timeline, durations, event_indicators, entries, ar=self.ar
            )

            yield self.returned_dict


class NonHomogeneousPoissonIterable(Iterable):
    """
    Iterable returning dict of 1D arrays selected from each iterations
    """

    reward_func = None
    discount_factor = None

    def __init__(
        self,
        size: int,
        tf: float,
        process: NonHomogeneousPoissonProcess,
        reward_func: Optional[
            Callable[[NDArray[np.float64]], NDArray[np.float64]]
        ] = None,
        discount_factor: Optional[
            Callable[[NDArray[np.float64]], NDArray[np.float64]]
        ] = None,
        t0: float = 0.0,
        ar: Optional[NDArray[np.float64]] = None,
        seed: Optional[int] = None,
        keep_last: bool = True,
    ):
        self.size = size
        self.tf = tf
        self.t0 = t0

        self.model = process.model
        self.model_args = process.model_args
        self.reward_func = reward_func
        self.discount_func = discount_factor
        self.seed = seed
        self.keep_last = keep_last
        self.ar = ar

        self.returned_dict = {
            "samples_ids": np.array([], dtype=np.int64),
            "assets_ids": np.array([], dtype=np.int64),
            "timeline": np.array([], dtype=np.float64),
            "durations": np.array([], dtype=np.float64),
            "a0": np.array([], dtype=np.float64),
            "af": np.array([], dtype=np.float64),
        }

    def compute_rewards(self, durations):
        return self.reward_func(durations)

    def update_returned_dict(
        self,
        timeline,
        durations,
        a0,
        af,
    ):
        # select values in t0 tf observation window
        if self.keep_last:
            timeline_selection = np.logical_and(
                self.t0 <= timeline, timeline <= self.tf
            )
        else:
            timeline_selection = np.logical_and(self.t0 <= timeline, timeline < self.tf)

        # get ids
        assets_ids, samples_ids = np.where(timeline_selection)

        self.returned_dict.update(
            samples_ids=samples_ids,
            assets_ids=assets_ids,
            timeline=timeline[timeline_selection],
            durations=durations[timeline_selection],
            a0=a0[timeline_selection],
            af=af[timeline_selection],
        )

    def __iter__(self):
        # construct iterator
        iterator = NonHomogeneousPoissonIterator(
            self.size, self.tf, self.t0, ar=self.ar, seed=self.seed
        )
        # set model in iterator
        iterator.load(self.model, self.model_args)
        ages = np.zeros_like(iterator.timeline)

        # yield dict of results until iterator stops
        for durations, a0, af in iterator:
            # select values in t0 tf observation window
            timeline = iterator.timeline
            self.update_returned_dict(timeline, durations, a0, af)
            yield self.returned_dict
