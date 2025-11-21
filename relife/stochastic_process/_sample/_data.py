from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class StochasticDataSample:
    time_window: tuple[float, float]
    struct_array: NDArray[np.void]

    @property
    def t0(self):
        return self.time_window[0]

    @property
    def tf(self):
        return self.time_window[1]

    def select(self, sample_id: Optional[int] = None, asset_id: Optional[int] = None) -> StochasticDataSample:
        mask: NDArray[np.bool_] = np.ones_like(self.struct_array, dtype=np.bool_)
        if sample_id is not None:
            mask = mask & np.isin(self.struct_array["sample_id"], sample_id)
        if asset_id is not None:
            mask = mask & np.isin(self.struct_array["asset_id"], asset_id)
        struct_subarray = self.struct_array[mask].copy()
        return replace(self, time_window=self.time_window, struct_array=struct_subarray)

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

    def get_matrix_events(self) -> NDArray[np.bool_]:
        index_in_rows = self.asset_id * self.nb_samples + self.sample_id
        unique_index, row_index = np.unique(index_in_rows, return_inverse=True)

        unique_timeline, col_timeline = np.unique(self.timeline, return_inverse=True)

        M = np.zeros((len(unique_index), len(unique_timeline)), dtype=bool)
        M[row_index, col_timeline] = self.event

        return M

    def get_matrix_preventive_renewals(self) -> NDArray[np.bool_]:
        index_in_rows = self.asset_id * self.nb_samples + self.sample_id
        unique_index, row_index = np.unique(index_in_rows, return_inverse=True)

        unique_timeline, col_timeline = np.unique(self.timeline, return_inverse=True)

        M = np.zeros((len(unique_index), len(unique_timeline)), dtype=bool)
        M[row_index, col_timeline] = ~self.event

        # Last observation isn't a renewal, end of observation
        M[:, -1] = False

        return M

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

    def get_timeline(self) -> NDArray[np.float64]:
        """
        Get the 1D array of timeline along the second axis for methods that return a 2D matrix
        """

        return np.unique(self.timeline)


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
        count_reward_matrix = np.zeros((self.nb_samples * self.nb_assets, time_linspace.shape[0]))

        for i, asset_id in enumerate(set(self.asset_id)):
            for j, sample_id in enumerate(set(self.sample_id)):
                index_row = i * self.nb_samples + j
                events_matrix = (
                    self.select(sample_id=sample_id, asset_id=asset_id).timeline.reshape(-1, 1) <= time_linspace
                )
                events_matrix *= self.select(sample_id=sample_id, asset_id=asset_id).reward.reshape(-1, 1)
                count_reward_matrix[index_row] = events_matrix.sum(axis=0)

        return count_reward_matrix
