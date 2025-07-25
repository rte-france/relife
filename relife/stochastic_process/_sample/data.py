from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class CountDataSample:
    t0: float
    tf: float
    struct_array: NDArray[np.void]

    def select(self, sample_id: Optional[int] = None, asset_id: Optional[int] = None) -> CountDataSample:
        mask: NDArray[np.bool_] = np.ones_like(self.struct_array, dtype=np.bool_)
        if sample_id is not None:
            mask = mask & np.isin(self.struct_array["sample_id"], sample_id)
        if asset_id is not None:
            mask = mask & np.isin(self.struct_array["asset_id"], asset_id)
        struct_subarray = self.struct_array[mask].copy()
        return replace(self, t0=self.t0, tf=self.tf, struct_array=struct_subarray)

    @property
    def time(self) -> NDArray[np.float64]:
        return self.struct_array["time"]

    @property
    def timeline(self) -> NDArray[np.float64]:
        return self.struct_array["timeline"]

    @property
    def sample_id(self) -> NDArray[np.uint32]:
        return self.struct_array["sample_id"]

    @property
    def asset_id(self) -> NDArray[np.uint32]:
        return self.struct_array["asset_id"]

    @property
    def nb_renewal(self) -> NDArray[np.uint32]:
        return self.struct_array["nb_renewal"]

    @property
    def event(self) -> NDArray[np.bool_]:
        return self.struct_array["event"]

    @property
    def entry(self) -> NDArray[np.float64]:
        return self.struct_array["entry"]


@dataclass
class RenewalProcessSample(CountDataSample):

    @staticmethod
    def _nb_events(selection: RenewalProcessSample) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sort = np.argsort(selection.struct_array["timeline"])
        timeline = selection.struct_array["timeline"][sort]
        counts = selection.struct_array["event"][sort]

        timeline = np.insert(timeline, 0, selection.t0)
        counts = np.insert(counts, 0, 0)
        counts[timeline == selection.tf] = 0
        return timeline, np.cumsum(counts)

    def nb_events(self, sample_id: int, asset_id: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        selection = self.select(sample_id=sample_id, asset_id=asset_id)
        return RenewalProcessSample._nb_events(selection)

    def mean_nb_events(self, asset_id: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        selection = self.select(asset_id=asset_id)
        timeline, counts = RenewalProcessSample._nb_events(selection)
        nb_sample = len(np.unique(selection.struct_array["sample_id"]))
        return timeline, counts / nb_sample


@dataclass
class RenewalRewardProcessSample(RenewalProcessSample):

    @staticmethod
    def _total_rewards(selection: RenewalRewardProcessSample) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sort = np.argsort(selection.struct_array["timeline"])
        timeline = selection.struct_array["timeline"][sort]
        reward = selection.struct_array["reward"][sort]

        timeline = np.insert(timeline, 0, selection.t0)
        reward = np.insert(reward, 0, 0)
        reward[timeline == selection.tf] = 0

        return timeline, np.cumsum(reward)

    def total_rewards(self, sample_id: int, asset_id: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        selection = self.select(sample_id=sample_id, asset_id=asset_id)
        return RenewalRewardProcessSample._total_rewards(selection)

    def mean_total_rewards(self, asset_id: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        selection = self.select(asset_id=asset_id)
        timeline, rewards = RenewalRewardProcessSample._nb_events(selection)
        nb_sample = len(np.unique(selection.struct_array["sample_id"]))
        return timeline, rewards / nb_sample


NonHomogeneousPoissonProcessSample = RenewalProcessSample
