from collections.abc import Mapping
from typing import Iterator, Optional, TypedDict

import numpy as np
from numpy.typing import NDArray

# this line goes in functions using the iterable and returning StochasticDataSample
# struct_array = np.concatenate(tuple(iterable))

# the symbols part of the public API of _data
__all__ = ["StochasticSampleMapping"]


# totality is True, all keys must be present
# beginning with _ because it is assumed to be internal to the module
class _StochasticSample(TypedDict):
    events: NDArray[np.bool_]
    preventive_renewals: NDArray[np.bool_]
    rewards: Optional[NDArray[np.float64]]


# Mapping is generic : you must specify value for KT, VT TypeVar
class StochasticSampleMapping(Mapping[str, NDArray[np.float64] | NDArray[np.bool_] | None]):

    # exposed data, part of the public object interface
    nb_assets: int
    nb_samples: int
    timeline: NDArray[np.float64]

    # hidden data for the internal object fonctionning
    _stochastic_data_sample: _StochasticSample

    def __init__(
        self,
        nb_assets: int,
        nb_samples: int,
        struct_array: NDArray[np.void],
    ) -> None:

        self.nb_assets = nb_assets
        self.nb_samples = nb_samples

        struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))
        # assets x samples are placed on axis 0
        index_in_rows = struct_array["asset_id"] * nb_samples + struct_array["sample_id"]
        _, row_index = np.unique(index_in_rows, return_inverse=True)

        # unique values of timeline on axis 1
        self.timeline, col_timeline = np.unique(struct_array["timeline"], return_inverse=True)

        # construction of internal _data
        events = np.zeros((self.nb_assets * self.nb_samples, self.timeline.size), dtype=bool)
        events[row_index, col_timeline] = struct_array["event"]
        preventive_renewals = np.zeros((self.nb_assets * self.nb_samples, self.timeline.size), dtype=bool)
        preventive_renewals[row_index, col_timeline] = ~struct_array["event"]
        preventive_renewals[:, -1] = False
        if "rewards" in struct_array:
            rewards = np.zeros((self.nb_assets * self.nb_samples, self.timeline.size), dtype=float)
            rewards[row_index, col_timeline] = struct_array["reward"]
        else:
            rewards = None
        self._stochastic_data_sample = {
            "events": events,
            "preventive_renewals": preventive_renewals,
            "rewards": rewards,
        }

    def __getitem__(self, key: str) -> NDArray[np.float64] | NDArray[np.bool_] | None:
        if key not in self._stochastic_data_sample:
            raise KeyError(f"Key {key} does not exists. Allowed keys are 'events' or 'preventive_renewals'")
        return self._stochastic_data_sample.get(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._stochastic_data_sample)

    def __len__(self) -> int:
        return len(self._stochastic_data_sample)

    # if select only changes _stochastic_data_sample, then get a deepcopy and change _stochastic_data_sample internally.

    # def select(
    #     self,
    #     sample_id: Optional[IntArrayLike] = None,
    #     asset_id: Optional[IntArrayLike] = None,
    # ) -> StochasticDataSample:
    #     """
    #     Focus on specific assets and samples. Return a subpart of StochasticDataSample.
    #     """

    #     if asset_id is None:
    #         asset_id = np.arange(self.nb_assets)
    #     if sample_id is None:
    #         sample_id = np.arange(self.nb_samples)

    #     asset_id = np.atleast_1d(asset_id)
    #     sample_id = np.atleast_1d(sample_id)

    #     new_nb_assets = asset_id.shape[0]
    #     new_nb_samples = sample_id.shape[0]

    #     mask = (asset_id[None, :] * self.nb_samples + sample_id[:, None]).flatten()

    #     new_data = {key: value[mask] for key, value in self._data.items()}

    #     return StochasticDataSample(
    #         timeline=self.timeline,
    #         nb_assets=new_nb_assets,
    #         nb_samples=new_nb_samples,
    #         data=new_data,
    #     )
