from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Optional, Sequence, Iterator

import numpy as np
from numpy.typing import NDArray


@dataclass
class CountData(ABC):
    samples_index: NDArray[np.int64] = field(repr=False)  # samples index
    assets_index: NDArray[np.int64] = field(repr=False)  # assets index
    order: NDArray[np.int64] = field(
        repr=False
    )  # order index (order in generation process)
    event_times: NDArray[np.float64] = field(repr=False)

    nb_samples: int = field(init=False)
    nb_assets: int = field(init=False)
    samples_unique_index: NDArray[np.int64] = field(
        init=False, repr=False
    )  # unique samples index
    assets_unique_index: NDArray[np.int64] = field(
        init=False, repr=False
    )  # unique assets index

    def __post_init__(self):
        fields_values = [
            getattr(self, _field.name) for _field in fields(self) if _field.init
        ]
        if not all(
            arr.ndim == 1 for arr in fields_values if isinstance(arr, np.ndarray)
        ):
            raise ValueError("All array values must be 1d")
        if (
            not len(
                set(
                    arr.shape[0] for arr in fields_values if isinstance(arr, np.ndarray)
                )
            )
            == 1
        ):
            raise ValueError("All array values must have the same shape")

        self.samples_unique_index = np.unique(
            self.samples_index
        )  # samples unique index
        self.assets_unique_index = np.unique(self.assets_index)  # assets unique index
        self.nb_samples = len(self.samples_unique_index)
        self.nb_assets = len(self.assets_unique_index)

    def __len__(self) -> int:
        return self.nb_samples * self.nb_assets

    def number_of_events(
        self, sample: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ind = self.samples_index == sample
        times = np.insert(np.sort(self.event_times[ind]), 0, 0)
        counts = np.arange(times.size)
        return times, counts

    def mean_number_of_events(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        times = np.insert(np.sort(self.event_times), 0, 0)
        counts = np.arange(times.size) / self.nb_samples
        return times, counts

    @abstractmethod
    def iter(self, sample: Optional[int] = None) -> "CountDataIterable": ...


class CountDataIterable:
    def __init__(self, data: CountData, field_names: Sequence[str]):
        """
        Parameters
        ----------
        data :
        field_names : fields iterate on
        """
        self.data = data
        sorted_index = np.lexsort((data.order, data.assets_index, data.samples_index))
        self.sorted_fields = tuple(
            (getattr(data, name)[sorted_index].copy() for name in field_names)
        )
        self.samples_index = data.samples_index[sorted_index].copy()
        self.assets_index = data.assets_index[sorted_index].copy()

    def __len__(self) -> int:
        return self.data.nb_samples * self.data.nb_assets

    def __iter__(self) -> Iterator[tuple[int, int, *tuple[NDArray[np.float64], ...]]]:

        for sample in self.data.samples_unique_index:
            sample_mask = self.samples_index == sample
            for asset in self.data.assets_unique_index:
                asset_mask = self.assets_index[sample_mask] == asset
                itervalues = tuple(
                    (v[sample_mask][asset_mask]) for v in self.sorted_fields
                )
                yield int(sample), int(asset), *itervalues
