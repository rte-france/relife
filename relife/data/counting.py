import copy
from dataclasses import dataclass, field, fields, replace
from typing import Optional, Union, NewType

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife.plots import (
    PlotCountingData,
    PlotRenewalData,
    PlotConstructor,
    PlotNHPPData,
)

Ids = NewType("Ids", Union[list[int, ...], tuple[int, ...], int])


@dataclass
class CountData:
    t0: float
    tf: float
    samples_ids: NDArray[np.int64] = field(repr=False)  # samples ids
    assets_ids: NDArray[np.int64] = field(repr=False)  # assets ids
    timeline: NDArray[np.float64] = field(repr=False)  # timeline (time of each events)
    nb_samples: int = field(init=False)
    nb_assets: int = field(init=False)
    samples_unique_ids: NDArray[np.int64] = field(
        init=False, repr=False
    )  # unique samples index
    assets_unique_ids: NDArray[np.int64] = field(
        init=False, repr=False
    )  # unique assets index

    def __post_init__(self):
        # sort fields

        sorted_indices = np.lexsort((self.timeline, self.assets_ids, self.samples_ids))

        self.samples_unique_ids = np.unique(self.samples_ids)
        self.assets_unique_ids = np.unique(self.assets_ids)

        self.nb_samples = len(self.samples_unique_ids)
        self.nb_assets = len(self.assets_unique_ids)

        for field_name, arr in self.fields():
            setattr(self, field_name, arr[sorted_indices])

    def fields(self):
        for field_def in fields(self):
            if field_def.init and field_def.name not in (
                "t0",
                "tf",
                "nb_samples",
                "nb_assets",
            ):
                yield field_def.name, getattr(self, field_def.name)

    def __len__(self) -> int:
        return self.nb_samples * self.nb_assets

    def select(self, sample: Optional[Ids] = None, asset: Optional[Ids] = None):
        # fluent interface to select sample

        selection = np.ones_like(self.timeline, dtype=np.bool_)

        if sample is not None:
            try:
                iter(sample)
            except TypeError:
                sample = (sample,)

            if not set(sample).issubset(self.samples_unique_ids):
                raise ValueError("Sample indices are not valid")

            selection = np.isin(self.samples_ids, sample)  # bool index

        if asset is not None:
            try:
                iter(asset)
            except TypeError:
                asset = (asset,)
            if not set(asset).issubset(self.assets_unique_ids):
                raise ValueError("Sample indices are not valid")

            selection = np.logical_and(np.isin(self.assets_ids, asset), selection)

        new_fields = {
            field_name: np.compress(selection, arr) for field_name, arr in self.fields()
        }

        return replace(copy.deepcopy(self), t0=self.t0, tf=self.tf, **new_fields)

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


@dataclass
class RenewalData(CountData):
    durations: NDArray[np.float64] = field(repr=False)
    rewards: NDArray[np.float64] = field(repr=False)

    def total_rewards(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sort = np.argsort(self.timeline)
        timeline = self.timeline[sort]
        rewards = self.rewards[sort]
        timeline = np.insert(timeline, 0, self.t0)
        rewards = np.insert(rewards, 0, 0)
        rewards[timeline == self.tf] = 0
        return timeline, rewards.cumsum()

    def mean_total_rewards(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline, rewards = self.total_rewards()
        return timeline, rewards / len(self)

    @property
    def plot(self) -> PlotConstructor:
        return PlotRenewalData(self)


@dataclass
class NHPPCountData(CountData):
    ages: NDArray[np.float64] = field(repr=False)
    events: NDArray[np.bool_] = field(repr=False)
    rewards: NDArray[np.float64] = field(repr=False)

    @override
    def nb_events(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sort = np.argsort(self.timeline)
        timeline = self.timeline[sort]
        timeline = np.insert(timeline, 0, self.t0)
        counts = self.events[sort].copy()
        counts = np.insert(counts, 0, 0)
        return timeline, np.cumsum(counts)

    def nb_repairs(self):
        return self.nb_events()

    def mean_nb_repairs(self):
        return self.mean_nb_events()

    def total_rewards(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sort = np.argsort(self.timeline)
        timeline = self.timeline[sort]
        rewards = self.rewards[sort]
        timeline = np.insert(timeline, 0, self.t0)
        rewards = np.insert(rewards, 0, 0)
        rewards[timeline == self.tf] = 0
        return timeline, rewards.cumsum()

    def mean_total_rewards(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline, rewards = self.total_rewards()
        return timeline, rewards / len(self)

    @property
    def plot(self) -> PlotConstructor:
        return PlotNHPPData(self)
