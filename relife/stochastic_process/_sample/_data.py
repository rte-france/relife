from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Union, List

import numpy as np
from numpy.typing import NDArray


@dataclass
class StochasticDataSample:
    time_window: tuple[float, float]
    nb_assets: int
    nb_samples: int
    _struct_array: NDArray[np.void]

    timeline : NDArray[np.float64] = None
    events: NDArray[np.bool_] = None
    preventive_renewals : NDArray[np.bool_] = None

    _row_index: NDArray[np.int64] = None
    _col_timeline: NDArray[np.int64] = None

    def __post_init__(self):

        # assets x samples are placed on axis 0
        index_in_rows = self._struct_array["asset_id"] * self.nb_samples + self._struct_array["sample_id"]
        unique_index, self._row_index = np.unique(index_in_rows, return_inverse=True)

        # unique values of timeline on axis 1
        self.timeline, self._col_timeline = np.unique(self._struct_array["timeline"], return_inverse=True)

        # 2D matrix of booleans to indicate observed events
        self.events = np.zeros(
            (self.nb_assets * self.nb_samples, self.timeline.size), dtype=bool
        )
        self.events[self._row_index, self._col_timeline] = self._struct_array["event"]

        # 2D matrix of booleans to indicate preventive renewals
        self.preventive_renewals = np.zeros(
            (self.nb_assets * self.nb_samples, self.timeline.size), dtype=bool
        )
        self.preventive_renewals[self._row_index, self._col_timeline] = ~self._struct_array["event"]
        self.preventive_renewals[:, -1] = False

    def _select_with_mask(self, mask)-> StochasticDataSample:
        """
        Take sub-matrix on specific assets and samples using a mask
        """
        return replace(self, events=self.events[mask], preventive_renewals=self.preventive_renewals[mask])

    def select(
        self, sample_id: Optional[Union[int|List[int]]] = None, asset_id: Optional[Union[int|List[int]]] = None
    ) -> StochasticDataSample:
        """
        Focus on specific assets and samples. Return a truncated StochasticDataSample.
        """

        if asset_id is None:
            asset_id = np.arange(self.nb_assets)
        if sample_id is None:
            sample_id = np.arange(self.nb_samples)

        sample_id = np.asarray(sample_id)
        asset_id = np.asarray(asset_id)

        nb_assets = 1 if (np.asarray(asset_id).ndim==0) else np.asarray(asset_id).shape[0]
        nb_samples = 1 if (np.asarray(sample_id).ndim==0) else np.asarray(asset_id).shape[0]
        
        mask = (asset_id[None,:]*self.nb_samples + sample_id[:,None]).flatten()
        
        return replace(self._select_with_mask(mask),nb_assets=nb_assets,nb_samples=nb_samples)
    
    def _select_from_struct(self, sample_id: Optional[Union[int|List[int]]] = None, asset_id: Optional[Union[int|List[int]]] = None):
        """
        Select the samples and assets in the struct array. Used for dev and tests only.
        """
        mask: NDArray[np.bool_] = np.ones_like(self._struct_array, dtype=np.bool_)
        if sample_id is not None:
            mask = mask & np.isin(self._struct_array["sample_id"], sample_id)
        if asset_id is not None:
            mask = mask & np.isin(self._struct_array["asset_id"], asset_id)
        return self._struct_array[mask].copy()



@dataclass
class StochasticRewardDataSample(StochasticDataSample):
    rewards: NDArray[np.float64] = None

    def __post__init__(self):
        super().__post__init__()

        self.rewards = np.zeros(
            (self.nb_assets * self.nb_samples, self.timeline.size), dtype=float
        )
        self.rewards[self._row_index,self._col_timeline] = self._struct_array["reward"]

    def _select_with_mask(self, mask)->StochasticRewardDataSample:
        return replace(self, events=self.events[mask], preventive_renewals=self.preventive_renewals[mask], rewards=self.rewards[mask])
