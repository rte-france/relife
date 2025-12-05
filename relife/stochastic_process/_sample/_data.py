from __future__ import annotations

from typing import Optional, Union, List
from collections.abc import Mapping

import numpy as np

from relife.stochastic_process._sample._iterables import StochasticDataIterable


def build_data_sample_from_iterable(
    iterable: StochasticDataIterable,
    nb_assets: int,
    nb_samples: int,
    time_window: tuple[float, float],
    is_reward: bool = False,
) -> StochasticDataSample:
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # assets x samples are placed on axis 0
    index_in_rows = struct_array["asset_id"] * nb_samples + struct_array["sample_id"]
    unique_index, row_index = np.unique(index_in_rows, return_inverse=True)

    # unique values of timeline on axis 1
    timeline, col_timeline = np.unique(struct_array["timeline"], return_inverse=True)

    zero_matrix = np.zeros((nb_assets * nb_samples, timeline.size), dtype=bool)

    # 2D matrix of booleans to indicate observed events
    events = zero_matrix.copy()
    events[row_index, col_timeline] = struct_array["event"]

    # 2D matrix of booleans to indicate preventive renewals
    preventive_renewals = zero_matrix.copy()
    preventive_renewals[row_index, col_timeline] = ~struct_array["event"]
    preventive_renewals[:, -1] = False

    data = {"events": events, "preventive_renewals": preventive_renewals}

    if is_reward:
        rewards = zero_matrix.copy().astype(float)
        rewards[row_index, col_timeline] = struct_array["reward"]
        data["rewards"] = rewards

    return StochasticDataSample(
        time_window=time_window, nb_assets=nb_assets, nb_samples=nb_samples, data=data
    )


class StochasticDataSample(Mapping):
    def __init__(
        self,
        time_window: tuple[float, float],
        nb_assets: int,
        nb_samples: int,
        data: dict,
    ):
        self.time_window = time_window
        self.nb_assets = nb_assets
        self.nb_samples = nb_samples
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def select(
        self,
        sample_id: Optional[Union[int | List[int]]] = None,
        asset_id: Optional[Union[int | List[int]]] = None,
    ) -> StochasticDataSample:
        if asset_id is None:
            asset_id = np.arange(self.nb_assets)
        if sample_id is None:
            sample_id = np.arange(self.nb_samples)

        asset_id = np.atleast_1d(asset_id)
        sample_id = np.atleast_1d(sample_id)

        new_nb_assets = asset_id.shape[0]
        new_nb_samples = sample_id.shape[0]

        mask = (asset_id[None, :] * self.nb_samples + sample_id[:, None]).flatten()

        new_data = {key: value[mask] for key, value in self._data.items()}

        return StochasticDataSample(
            time_window=self.time_window,
            nb_assets=new_nb_assets,
            nb_samples=new_nb_samples,
            data=new_data,
        )
