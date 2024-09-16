from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from relife2.core import ParametricModel, LifetimeModel

T = TypeVar("T")


@dataclass(frozen=True)
class SampleData:
    values: NDArray[np.float64]
    samples_index: NDArray[np.int64]
    assets_index: NDArray[np.int64]


class SampleIterator(Iterator[SampleData], ABC):
    """Iterator pattern expecting nb of samples and optionally a nb of assets"""

    def __init__(
        self,
        model: ParametricModel,
        nb_samples: int,
        nb_assets: int = 1,
        args: tuple[NDArray[np.float64], ...] = (),
    ):
        if bool(args) and args[0].ndim == 2:
            if nb_assets != 1 and nb_assets != args[0].shape[0]:
                raise ValueError
            nb_assets = args[0].shape[0]
            self.size = nb_samples
        else:
            self.size = nb_samples * nb_assets
        self.nb_samples = nb_samples
        self.nb_assets = nb_assets
        self.model = model
        self.args = args

    @abstractmethod
    def __next__(self) -> SampleData: ...


# concrete SampleIterator
class LifetimeSampleIterator(SampleIterator):
    def __init__(
        self,
        model: LifetimeModel,
        size: int,
        end_time: float,
        nb_assets: int = 1,
        args: tuple[NDArray[np.float64], ...] = (),
    ):
        super().__init__(model, size, nb_assets, args)

        self.end_time = end_time
        self.spent_time = np.zeros(self.nb_samples * self.nb_assets)
        self.samples_index, self.assets_index = np.unravel_index(
            np.arange(self.nb_samples * self.nb_assets),
            (self.nb_samples, self.nb_assets),
        )

    def __iter__(self):
        return self

    def __next__(
        self,
    ) -> SampleData:
        # shape : (nb_assets * nb_samples)
        still_valid = self.spent_time < self.end_time
        if still_valid.any():
            # shape : (nb_assets, nb_samples)
            event_times = self.model.rvs(*self.args, size=self.size).reshape(-1)[
                still_valid
            ]
            self.spent_time[still_valid] += event_times
            return SampleData(
                event_times,
                self.samples_index[still_valid],
                self.assets_index[still_valid],
            )
        else:
            raise StopIteration


# generic aggregator function of SampleData
def aggregate(iterator: SampleIterator) -> SampleData:
    """function that aggregates results of EventIterator iterations in arrays, sorted by samples_ids and assets_ids"""
    values = np.array([], dtype=np.float64)
    samples_index = np.array([], dtype=np.int64)
    assets_index = np.array([], dtype=np.int64)
    for event in iterator:
        values = np.concatenate((values, event.values))
        samples_index = np.concatenate((samples_index, event.samples_index))
        assets_index = np.concatenate((assets_index, event.assets_index))
    sorted_index = np.lexsort((assets_index, samples_index))
    values = values[sorted_index]
    samples_index = samples_index[sorted_index]
    assets_index = assets_index[sorted_index]
    return SampleData(values, samples_index, assets_index)


# generic SampleIterable class
class SampleIterable:

    def __init__(self, data: SampleData):
        self.data = data
        self.nb_samples = len(
            np.unique(self.data.samples_index, return_counts=True)[-1]
        )
        self.nb_assets = len(np.unique(self.data.assets_index, return_counts=True)[-1])
        self.flatten_samples_index = (
            np.where(self.data.samples_index[:-1] != self.data.samples_index[1:])[0] + 1
        )
        # self.assets_partitions = (
        #     np.where(self.data.assets_index[:-1] != self.data.assets_index[1:])[0] + 1
        # )

    def __len__(self):
        if self.nb_assets == 1:
            return self.nb_samples
        else:
            return self.nb_assets

    @property
    def mean_number_of_events(self):
        return np.mean(
            np.diff(
                self.flatten_samples_index,
                prepend=0,
                append=len(self.data.values) - 1,
            )
        )

    def __get(self, index: int):
        if self.flatten_samples_index.size == 0:
            slice_index = slice(None, None)
        else:
            if index == len(self.flatten_samples_index):
                start = self.flatten_samples_index[-1]
            elif index > len(self.flatten_samples_index):
                raise IndexError
            else:
                start = self.flatten_samples_index[index]
            if start == self.flatten_samples_index[0]:
                slice_index = slice(None, start)
            elif start != self.flatten_samples_index[-1]:
                stop = self.flatten_samples_index[index + 1]
                slice_index = slice(start, stop)
            else:
                slice_index = slice(start, None)
        return self.data.values[slice_index], self.data.assets_index[slice_index]

    def __getitem__(self, index: int):
        try:
            values, assets_index = self.__get(index)
        except IndexError as err:
            raise IndexError(
                f"index {index} is out of bounds for {self.nb_samples} samples on {self.nb_assets} assets"
            ) from err
        if self.nb_assets == 1:
            nb_events = len(values)
        else:
            values = np.split(
                values, np.where(assets_index[:-1] != assets_index[1:])[0] + 1
            )
            nb_events = list(map(len, values))
        return {"values": values, "nb_events": nb_events}


def sample_lifetimes(
    model: LifetimeModel,
    size: int,
    end_time: float,
    nb_assets: int = 1,
    args: tuple[NDArray[np.float64], ...] = (),
):
    return SampleIterable(
        aggregate(LifetimeSampleIterator(model, size, end_time, nb_assets, args))
    )
