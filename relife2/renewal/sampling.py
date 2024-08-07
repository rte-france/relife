from collections.abc import Sequence

import numpy as np

from relife2.functions.distributions import DistributionFunctions
from relife2.functions.regressions import RegressionFunctions
from .iterators import DistributionIterator, EventIterator, RegressionIterator
from ..functions.core import ParametricLifetimeFunctions


def event_aggregator(iterator: EventIterator):
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


class EventSampling(Sequence):
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

    def _get_values_from_partitions(self, index: int):
        if self.nb_assets == 1:
            partitions = self._samples_partitions
        else:
            partitions = self._assets_partitions
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

    def __getitem__(self, index: int):
        """
        If nb_assets was given, item corresponds to sampler number
        """
        if not isinstance(index, int):
            raise ValueError(f"index {index} must be int type")
        if self.nb_assets == 1:
            try:
                values, _ = self._get_values_from_partitions(index)
            except IndexError as err:
                raise IndexError(
                    f"index {index} is out of bounds for {self.nb_samples} samples on {self.nb_assets} assets"
                ) from err
            nb_events = len(values)
        else:
            try:
                values_of_asset, values_index = self._get_values_from_partitions(index)
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


class DistributionSampling(EventSampling):
    """Concrete EventSampling for DistributionFunctions"""

    def __init__(self, functions, nb_samples, end_time, nb_assets=1):
        values, samples_ids, assets_ids = event_aggregator(
            DistributionIterator(functions, nb_samples, end_time, nb_assets=nb_assets)
        )
        super().__init__(values, samples_ids, assets_ids)


class RegressionSampling(EventSampling):
    """Concrete EventSampling for RegressionFunctions"""

    def __init__(self, functions, covar, nb_samples, end_time):
        values, samples_ids, assets_ids = event_aggregator(
            RegressionIterator(functions, covar, nb_samples, end_time)
        )
        super().__init__(values, samples_ids, assets_ids)


def sample(functions: ParametricLifetimeFunctions, *args, **kwargs) -> EventSampling:
    """Factory of EventSampling sequence object"""
    if isinstance(functions, DistributionFunctions):
        return DistributionSampling(functions, *args, **kwargs)
    elif isinstance(functions, RegressionFunctions):
        return RegressionSampling(functions, *args, **kwargs)
    else:
        raise ValueError


# distribution_model = Weibull(2.0, 0.1)
# regression_model = AFT(distribution_model, (0.3, 0.8))
#
# distribution_sampling = sampling(distribution_model.functions, 3, 50)
# regression_sampling = sampling(
#     regression_model.functions, np.array([[0.1, 0.2], [0.5, 0.8], [0.3, 0.6]]), 3, 50
# )
#
# print("nb of samples:", len(distribution_sampling))
# print("lifetimes for sample 1:", distribution_sampling[0]["values"])
# print("lifetimes for sample 3:", distribution_sampling[2]["values"])
# print("nb of events for sample 3:", distribution_sampling[2]["nb_events"])
# print("mean number of events", distribution_sampling.mean_number_of_events)
#
# print("nb of samples:", len(regression_sampling))
# print("lifetimes for asset 1 sample 1:", regression_sampling[0]["values"][0])
# print("lifetimes for asset 1 :", regression_sampling[0]["values"])
# print("lifetimes for asset 1 sample 3:", regression_sampling[0]["values"][2])
# print("nb of events for asset 2 sample 3:", regression_sampling[1]["nb_events"][2])
# print("mean number of events", regression_sampling.mean_number_of_events)
