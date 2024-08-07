from abc import ABC
from collections.abc import Sequence

import numpy as np

from .iterators import EventIterator


def aggregate_events(iterator: EventIterator):
    """function that aggregates results of EventIterator iterations in arrays, sorted by samples_ids and assets_ids"""
    values = np.array([], dtype=np.float64)
    samples_ids = np.array([], dtype=np.float64)
    assets_ids = np.array([], dtype=np.float64)
    for _values, _samples_ids, _assets_ids in iterator:
        values = np.concatenate((values, _values))
        samples_ids = np.concatenate((samples_ids, _samples_ids))
        assets_ids = np.concatenate((assets_ids, _assets_ids))
    sorted_index = np.lexsort((samples_ids, assets_ids))
    values = values[sorted_index]
    samples_ids = samples_ids[sorted_index]
    assets_ids = assets_ids[sorted_index]
    return values, samples_ids, assets_ids


class EventSampling(Sequence, ABC):
    """EventSampling Sequence object that stores results of sample"""

    def __init__(self, values, samples_ids, assets_ids):
        self.values = values
        self.samples_ids = samples_ids
        self.assets_ids = assets_ids

        self.nb_samples = len(np.unique(self.samples_ids, return_counts=True)[-1])
        self.nb_assets = len(np.unique(self.assets_ids, return_counts=True)[-1])

        self._samples_partitions = (
            np.where(self.samples_ids[:-1] != self.samples_ids[1:])[0] + 1
        )
        self._assets_partitions = (
            np.where(self.assets_ids[:-1] != self.assets_ids[1:])[0] + 1
        )

    def __len__(self):
        if self.nb_assets == 1:
            return self.nb_samples
        else:
            return self.nb_assets

    @property
    def mean_number_of_events(self):
        return np.mean(
            np.diff(self._samples_partitions, prepend=0, append=len(self.values) - 1)
        )

    def _get_values_from_partitions(self, partitions: np.ndarray, index: int):
        if index == len(partitions):
            start = partitions[-1]
        elif index > len(partitions):
            raise IndexError
        else:
            start = partitions[index]
        if start == partitions[0]:
            values_index = slice(None, start)
        elif start != partitions[-1]:
            stop = partitions[index + 1]
            values_index = slice(start, stop)
        else:
            values_index = slice(start, None)
        return self.values[values_index], values_index


class SamplingWithoutAssets(EventSampling):
    def __getitem__(self, index: int):
        try:
            values, _ = self._get_values_from_partitions(
                self._samples_partitions, index
            )
        except IndexError as err:
            raise IndexError(
                f"index {index} is out of bounds for {self.nb_samples} samples on {self.nb_assets} assets"
            ) from err
        nb_events = len(values)
        return {"values": values, "nb_events": nb_events}


class SamplingWithAssets(EventSampling):
    def __getitem__(self, index: int):
        try:
            values_of_asset, values_index = self._get_values_from_partitions(
                self._assets_partitions, index
            )
        except IndexError as err:
            raise IndexError(
                f"index {index} is out of bounds for {self.nb_samples} samples on {self.nb_assets} assets"
            ) from err
        samples_ids = self.samples_ids[values_index]
        values = np.split(
            values_of_asset, np.where(samples_ids[:-1] != samples_ids[1:])[0] + 1
        )
        nb_events = list(map(len, values))
        return {"values": values, "nb_events": nb_events}
