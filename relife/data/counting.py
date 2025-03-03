import copy
from abc import ABC
from dataclasses import dataclass, field, fields, replace
from typing import Optional, Union, NewType

import numpy as np
from numpy.typing import NDArray


Ids = NewType("Ids", Union[list[int, ...], tuple[int, ...], int])


@dataclass
class CountData(ABC):
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

        for field_def in fields(self):
            if field_def.init and field_def.name not in ("t0", "tf"):
                arr = getattr(self, field_def.name)
                setattr(self, field_def.name, arr[sorted_indices])

    def __len__(self) -> int:
        return self.nb_samples * self.nb_assets

    def select(self, sample: Optional[Ids] = None, asset: Optional[Ids] = None):
        # fluent interface to select sample

        sample_selection = slice(None, None)
        asset_selection = slice(None, None)

        if sample is not None:
            try:
                iter(sample)
            except TypeError:
                sample = (sample,)

            if not set(sample).issubset(self.samples_unique_ids):
                raise ValueError("Sample indices are not valid")

            print(np.isin(self.samples_ids, sample))
            print(np.nonzero(np.isin(self.samples_ids, sample))[0])

            sample_selection = np.nonzero(np.isin(self.samples_ids, sample))[0]

        if asset is not None:
            try:
                iter(asset)
            except TypeError:
                asset = (asset,)
            if not set(asset).issubset(self.assets_unique_ids):
                raise ValueError("Sample indices are not valid")
            asset_selection = np.nonzero(
                np.isin(self.assets_ids[sample_selection], asset)
            )[0]

        print(sample_selection)
        print(asset_selection)

        new_fields = {
            field_def.name: getattr(self, field_def.name)[sample_selection][
                asset_selection
            ]
            for field_def in fields(self)
            if field_def.name not in ("t0", "tf")
        }

        return replace(copy.deepcopy(self), t0=self.t0, tf=self.tf, **new_fields)

    def number_of_events(
        self, sample_id: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ind = self.samples_ids == sample_id
        times = np.insert(np.sort(self.timeline[ind]), 0, 0)
        counts = np.arange(times.size)
        return times, counts

    def mean_number_of_events(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        times = np.insert(np.sort(self.timeline), 0, 0)
        counts = np.arange(times.size) / self.nb_samples
        return times, counts

    # @abstractmethod
    # def iter(self, sample_id: Optional[int] = None) -> "CountDataIterable":
    #     """
    #     Iterate over count data.
    #
    #     Provide an iterable access pattern for count-based data, optionally restricted by a provided sample size.
    #
    #     Parameters:
    #         sample_id: int (optional)
    #             Unique sample identifier.
    #     Returns:
    #         CountDataIterable:
    #             An iterable object that facilitates iteration over the count data.
    #     """
    #     ...


# class CountDataIterable:
#     def __init__(self, data: CountData, field_names: Sequence[str]):
#         """
#         Parameters
#         ----------
#         data :
#         field_names : fields iterate on
#         """
#         self.data = data
#         self.sorted_indices = np.lexsort(
#             (data.timeline, data.assets_ids, data.samples_ids)
#         )
#         self.sorted_fields = {
#             name: getattr(data, name)[self.sorted_indices].copy()
#             for name in field_names
#         }
#         self.samples_index = data.samples_ids[self.sorted_indices].copy()
#         self.assets_index = data.assets_ids[self.sorted_indices].copy()
#
#     def __len__(self) -> int:
#         return self.data.nb_samples * self.data.nb_assets
#
#     def __iter__(self) -> Iterator[tuple[int, int, *tuple[NDArray[np.float64], ...]]]:
#         """
#         Iterator over the CountData, yielding tuples with
#         (sample_id, asset_id, and corresponding field values).
#
#         Yields
#         ------
#         tuple[int, int, *tuple[NDArray[np.float64], ...]]
#         """
#         for sample_id in self.data.samples_unique_ids:
#             sample_specific_mask = self.samples_index == sample_id
#             for asset_id in self.data.assets_unique_ids:
#                 yield self._get_sample_asset_values(
#                     sample_id, asset_id, sample_specific_mask
#                 )
#
#     def _get_sample_asset_values(
#         self, sample_id: int, asset_id: int, sample_mask: NDArray[np.bool_]
#     ) -> tuple[int, int, *tuple[NDArray[np.float64], ...]]:
#         """
#         Extract values for a specific sample and asset combination.
#
#         Parameters
#         ----------
#         sample_id : int
#             Unique sample identifier.
#         asset_id : int
#             Unique asset identifier.
#         sample_mask : NDArray[np.bool_]
#             Mask indicating which entries belong to the sample.
#
#         Returns
#         -------
#         tuple[int, int, *tuple[NDArray[np.float64], ...]]
#         """
#         asset_specific_mask = self.assets_index[sample_mask] == asset_id
#         field_values_in_scope = {
#             k: v[sample_mask][asset_specific_mask]
#             for k, v in self.sorted_fields.items()
#         }
#         return int(sample_id), int(asset_id), field_values_in_scope
