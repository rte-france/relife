# pyright: basic

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterator, Sequence, TypedDict

import numpy as np
from numpy.typing import NDArray

__all__ = ["StochasticSampleMapping"]


class StochasticSample(TypedDict):
    """
    Container for 2D matrixes of stochastic samples.
    Assets x Samples on axis 0 and Timeline on axis 1.
    Values are either booleans to indicate events and renewals, or floats for rewards.
    """

    events: NDArray[np.bool_]
    preventive_renewals: NDArray[np.bool_]
    rewards: NDArray[np.float64] | None


class StochasticSampleMapping(
    Mapping[str, NDArray[np.float64] | NDArray[np.bool_] | None]
):
    """
    Mapping class to manipulate Stochastic samples.
    Access to 2D matrixes of events, renewals and rewards and method for selecting sub-samples of the full data.
    """

    nb_assets: int
    nb_samples: int
    timeline: NDArray[np.float64]

    _stochastic_data_sample: StochasticSample

    def __init__(
        self,
        nb_assets: int,
        nb_samples: int,
        timeline: NDArray[np.float64],
        stochastic_data_sample: StochasticSample,
    ) -> None:
        self.nb_assets = nb_assets
        self.nb_samples = nb_samples
        self.timeline = timeline
        self._stochastic_data_sample = stochastic_data_sample

    @classmethod
    def from_struct_array(
        cls, struct_array: NDArray[np.void], nb_assets: int, nb_samples: int
    ) -> StochasticSampleMapping:
        """
        Class method to build the _StochasticSample and timeline from a struct_array before initiating.
        """

        # assets x samples are placed on axis 0
        index_in_rows = (
            struct_array["asset_id"] * nb_samples + struct_array["sample_id"]
        )
        _, row_index = np.unique(index_in_rows, return_inverse=True)

        # unique values of timeline on axis 1
        timeline, col_timeline = np.unique(
            struct_array["timeline"], return_inverse=True
        )

        # construction of matrixes
        events = np.zeros((nb_assets * nb_samples, timeline.size), dtype=bool)
        events[row_index, col_timeline] = struct_array["event"]
        preventive_renewals = np.zeros(
            (nb_assets * nb_samples, timeline.size), dtype=bool
        )
        preventive_renewals[row_index, col_timeline] = ~struct_array["event"]
        preventive_renewals[:, -1] = False

        if "rewards" in struct_array.dtype.fields.keys():
            rewards = np.zeros((nb_assets * nb_samples, timeline.size), dtype=float)
            rewards[row_index, col_timeline] = struct_array["reward"]
        else:
            rewards = None

        stochastic_data_sample = {
            "events": events,
            "preventive_renewals": preventive_renewals,
            "rewards": rewards,
        }

        return cls(
            nb_assets=nb_assets,
            nb_samples=nb_samples,
            timeline=timeline,
            stochastic_data_sample=stochastic_data_sample,
        )

    def __getitem__(self, key: str) -> NDArray[np.float64] | NDArray[np.bool_] | None:
        if key not in self._stochastic_data_sample:
            raise KeyError(
                f"Key {key} does not exists. Allowed keys are 'events', 'preventive_renewals' or 'rewards'"
            )
        return self._stochastic_data_sample.get(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._stochastic_data_sample)

    def __len__(self) -> int:
        return len(self._stochastic_data_sample)

    def select(
        self,
        sample_id: int | Sequence[int] | None = None,
        asset_id: int | Sequence[int] | None = None,
    ) -> StochasticSampleMapping:
        """
        Select a sub part of the sample based on assets and samples ids.
        Returns a truncated StochasticSampleMapping by selecting the corresponding rows in the 2D matrixes.
        """
        if asset_id is None:
            asset_id = np.arange(self.nb_assets)
        if sample_id is None:
            sample_id = np.arange(self.nb_samples)

        asset_id = np.atleast_1d(asset_id)
        sample_id = np.atleast_1d(sample_id)

        new_nb_assets = asset_id.shape[0]
        new_nb_samples = sample_id.shape[0]

        mask = (asset_id[None, :] * self.nb_samples + sample_id[:, None]).flatten()

        new_stochastic_data_sample = {
            key: value[mask] for key, value in self._stochastic_data_sample.items()
        }

        return StochasticSampleMapping(
            nb_assets=new_nb_assets,
            nb_samples=new_nb_samples,
            timeline=self.timeline,
            stochastic_data_sample=new_stochastic_data_sample,
        )
