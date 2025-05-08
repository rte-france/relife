from typing import Optional, Self

import numpy as np
from numpy.typing import NDArray, ArrayLike

from relife._plots import (
    PlotConstructor,
    PlotCountingData,
)
from .iterators import RenewalProcessIterator

class RenewalProcessSample:
    def __init__(self, iterator : RenewalProcessIterator):
        struct_array = np.concatenate(tuple((arr for arr in iterator)))
        self._struct_array = np.sort(struct_array, order=("sample_id", "asset_id", "timeline"))
        self.t0 = iterator.t0
        self.tf = iterator.tf
        self.nb_samples = len(np.unique(self._struct_array["sample_id"]))
        self.nb_assets = len(np.unique(self._struct_array["asset_id"]))

    @property
    def timeline(self) -> NDArray[np.float64]:
        return self._struct_array["timeline"]

    @property
    def fields(self) -> tuple[str, ...]:
        return self._struct_array.dtype.names

    def __len__(self) -> int:
        return len(self._struct_array)

    def select(self, sample_id: Optional[ArrayLike] = None, asset_id: Optional[ArrayLike] = None) -> Self:
        mask = np.ones(len(self._struct_array), dtype=np.bool_)
        if sample_id is not None:
            mask = mask & np.isin(self._struct_array['sample_id'], sample_id)
        if asset_id is not None:
            mask = mask & np.isin(self._struct_array['asset_id'], asset_id)
        substruct_array = self._struct_array[mask].copy()
        return RenewalProcessSample(iter((substruct_array,))) # iterator that returns one struct_array

    def nb_events(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sort = np.argsort(self.timeline)
        timeline = self.timeline[sort]
        counts = np.ones_like(timeline)
        timeline = np.insert(timeline, 0, self.t0)
        counts = np.insert(counts, 0, 0)
        counts[timeline == self.tf] = 0
        return timeline, np.cumsum(counts)

    def mean_nb_events(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline, counts = self.nb_events()
        return timeline, counts / len(self)

    @property
    def plot(self) -> PlotConstructor:
        return PlotCountingData(self)
#
# @dataclass
# class RenewalRewardProcessSample(RenewalProcessSample):
#
#
#
#
# @dataclass
# class RenewalData(CountData):
#     durations: NDArray[np.float64] = field(repr=False)
#     rewards: NDArray[np.float64] = field(repr=False)
#
#     def total_rewards(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#         sort = np.argsort(self.timeline)
#         timeline = self.timeline[sort]
#         rewards = self.rewards[sort]
#         timeline = np.insert(timeline, 0, self.t0)
#         rewards = np.insert(rewards, 0, 0)
#         rewards[timeline == self.tf] = 0
#         return timeline, rewards.cumsum()
#
#     def mean_total_rewards(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#         timeline, rewards = self.total_rewards()
#         return timeline, rewards / len(self)
#
#     @property
#     def plot(self) -> PlotConstructor:
#         return PlotRenewalData(self)
#
#
# @dataclass
# class NHPPCountData(CountData):
#     ages: NDArray[np.float64] = field(repr=False)
#     events_indicators: NDArray[np.bool_] = field(repr=False)
#     rewards: NDArray[np.float64] = field(repr=False)
#
#     @override
#     def nb_events(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#         sort = np.argsort(self.timeline)
#         timeline = self.timeline[sort]
#         timeline = np.insert(timeline, 0, self.t0)
#         counts = self.events_indicators[sort].copy()
#         counts = np.insert(counts, 0, 0)
#         return timeline, np.cumsum(counts)
#
#     def nb_repairs(self):
#         return self.nb_events()
#
#     def mean_nb_repairs(self):
#         return self.mean_nb_events()
#
#     def total_rewards(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#         sort = np.argsort(self.timeline)
#         timeline = self.timeline[sort]
#         rewards = self.rewards[sort]
#         timeline = np.insert(timeline, 0, self.t0)
#         rewards = np.insert(rewards, 0, 0)
#         rewards[timeline == self.tf] = 0
#         return timeline, rewards.cumsum()
#
#     def mean_total_rewards(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#         timeline, rewards = self.total_rewards()
#         return timeline, rewards / len(self)
#
#     @property
#     def plot(self) -> PlotConstructor:
#         return PlotNHPPData(self)
