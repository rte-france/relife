from abc import ABC
from collections.abc import Iterator

import numpy as np

from relife2.functions import DistributionFunctions, RegressionFunctions


class EventIterator(Iterator, ABC):
    """Iterator pattern expecting nb of samples and optionally a nb of assets"""

    def __init__(self, functions, nb_samples, nb_assets=1):
        self.functions = functions
        self.nb_samples = nb_samples
        self.nb_assets = nb_assets


class DistributionIterator(EventIterator):
    """Concrete EventIterator for DistributionFunctions"""

    def __init__(self, functions: DistributionFunctions, nb_samples, end_time):
        super().__init__(functions, nb_samples)
        self.end_time = end_time
        self.durations = np.zeros(self.nb_samples)

    def __next__(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        unfailed_samples = self.durations < self.end_time
        if unfailed_samples.any():
            rvs_size = np.sum(unfailed_samples)
            event_times = self.functions.rvs(size=rvs_size).reshape(-1)
            self.durations[unfailed_samples] += event_times
            samples_ids = np.where(unfailed_samples)[0]
            assets_ids = np.zeros_like(samples_ids)
            return event_times, samples_ids, assets_ids
        else:
            raise StopIteration


class RegressionIterator(EventIterator):
    """Concrete EventIterator for RegressionFunctions"""

    def __init__(self, functions: RegressionFunctions, covar, nb_samples, end_time):
        super().__init__(functions, nb_samples, nb_assets=covar.shape[0])
        self.end_time = end_time
        self.durations = np.zeros(self.nb_assets * self.nb_samples)
        self.functions.covar = covar

    def __next__(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        unfailed_samples = (
            self.durations < self.end_time
        )  # shape : (nb_assets * nb_samples)
        if unfailed_samples.any():
            # shape : (nb_assets, nb_samples)
            event_times = self.functions.rvs(size=self.nb_samples).reshape(-1)[
                unfailed_samples
            ]
            self.durations[unfailed_samples] += event_times
            samples_ids = np.where(unfailed_samples)[0] % self.nb_assets
            assets_ids = np.where(unfailed_samples)[0] // self.nb_assets
            return event_times, samples_ids, assets_ids
        else:
            raise StopIteration
