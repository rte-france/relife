from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class StochasticDataSample:
    t0: float
    tf: float
    struct_array: NDArray[np.void]

    def select(
        self, sample_id: Optional[int] = None, asset_id: Optional[int] = None
    ) -> StochasticDataSample:
        mask: NDArray[np.bool_] = np.ones_like(self.struct_array, dtype=np.bool_)
        if sample_id is not None:
            mask = mask & np.isin(self.struct_array["sample_id"], sample_id)
        if asset_id is not None:
            mask = mask & np.isin(self.struct_array["asset_id"], asset_id)
        struct_subarray = self.struct_array[mask].copy()
        return replace(self, t0=self.t0, tf=self.tf, struct_array=struct_subarray)

    @property
    def nb_assets(self) -> int:
        return len(set(self.struct_array["asset_id"]))

    @property
    def nb_samples(self) -> int:
        return len(set(self.struct_array["sample_id"]))

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
    def event(self) -> NDArray[np.bool_]:
        return self.struct_array["event"]

    @property
    def entry(self) -> NDArray[np.float64]:
        return self.struct_array["entry"]

    def get_count_observation(self, tf: float, nb_steps: int) -> NDArray[np.int64]:
        """
        Count the increasing number of observations made for each asset x sample  on a given time range.
        Build a 2D Matrix with samples and assets on first axis and observation times in second axis.
        """

        return self.get_count_event(tf, nb_steps) + self.get_count_preventive_renewal(
            tf, nb_steps
        )

    def get_count_event(self, tf: float, nb_steps: int) -> NDArray[np.int64]:
        """
        Count the increasing number of observed events for each asset x sample  on a given time range.
        Build a 2D Matrix with samples and assets on first axis and observation times in second axis.
        """
        time_linspace = np.linspace(0, tf, nb_steps)
        count_event_matrix = np.zeros(
            (self.nb_samples * self.nb_assets, time_linspace.shape[0])
        )

        for i, asset_id in enumerate(set(self.asset_id)):
            for j, sample_id in enumerate(set(self.sample_id)):
                index_row = i * self.nb_samples + j
                events_matrix = (
                    self.select(
                        sample_id=sample_id, asset_id=asset_id
                    ).timeline.reshape(-1, 1)
                    <= time_linspace
                )
                events_matrix *= self.select(
                    sample_id=sample_id, asset_id=asset_id
                ).event.reshape(-1, 1)
                count_event_matrix[index_row] = events_matrix.sum(axis=0)

        return count_event_matrix

    def get_count_preventive_renewal(
        self, tf: float, nb_steps: int
    ) -> NDArray[np.int64]:
        """
        Count the increasing number of preventive renewal made for each asset x sample  on a given time range.
        Build a 2D Matrix with samples and assets on first axis and observation times in second axis.
        """
        time_linspace = np.linspace(0, tf, nb_steps)
        count_preventive_renewal_matrix = np.zeros(
            (self.nb_samples * self.nb_assets, time_linspace.shape[0])
        )

        for i, asset_id in enumerate(set(self.asset_id)):
            for j, sample_id in enumerate(set(self.sample_id)):
                index_row = i * self.nb_samples + j
                events_matrix = (
                    self.select(
                        sample_id=sample_id, asset_id=asset_id
                    ).timeline.reshape(-1, 1)
                    <= time_linspace
                )
                events_matrix *= ~self.select(
                    sample_id=sample_id, asset_id=asset_id
                ).event.reshape(-1, 1)
                count_preventive_renewal_matrix[index_row] = events_matrix.sum(axis=0)

        count_preventive_renewal_matrix[:, -1] -= (
            1  # end of observation window isn't a renewal
        )
        return count_preventive_renewal_matrix

    def get_sample_ids(self) -> NDArray[np.int64]:
        """
        Get the 1D array of sample ids along the first axis for methods that return a 2D matrix
        """

        return np.array(list(set(self.sample_id)) * self.nb_assets)

    def get_asset_ids(self) -> NDArray[np.int64]:
        """
        Get the 1D array of asset ids along the first axis for methods that return a 2D matrix
        """

        return np.repeat(np.array(list(set(self.asset_id))), self.nb_samples)


@dataclass
class StochasticRewardDataSample(StochasticDataSample):
    @property
    def reward(self) -> NDArray[np.float64]:
        return self.struct_array["reward"]

    def get_count_reward(self, tf: float, nb_steps: int) -> NDArray[np.float64]:
        """
        Count the increasing total reward for each asset x sample  on a given time range.
        Build a 2D Matrix with samples and assets on first axis and observation times in second axis.
        """
        time_linspace = np.linspace(0, tf, nb_steps)
        count_reward_matrix = np.zeros(
            (self.nb_samples * self.nb_assets, time_linspace.shape[0])
        )

        for i, asset_id in enumerate(set(self.asset_id)):
            for j, sample_id in enumerate(set(self.sample_id)):
                index_row = i * self.nb_samples + j
                events_matrix = (
                    self.select(
                        sample_id=sample_id, asset_id=asset_id
                    ).timeline.reshape(-1, 1)
                    <= time_linspace
                )
                events_matrix *= self.select(
                    sample_id=sample_id, asset_id=asset_id
                ).reward.reshape(-1, 1)
                count_reward_matrix[index_row] = events_matrix.sum(axis=0)

        return count_reward_matrix
