from abc import ABC
from collections.abc import Iterator

import numpy as np

from relife2.fiability import Distribution, Regression


class EventIterator(Iterator, ABC):
    """Iterator pattern expecting nb of samples and optionally a nb of assets"""

    def __init__(self, model, nb_samples, nb_assets):
        self.model = model
        self.nb_samples = nb_samples
        self.nb_assets = nb_assets


def transform_1d_index(index, nb_samples, nb_assets):
    if 1 < nb_assets < nb_samples:
        samples_ids = np.where(index)[0] % nb_assets
        assets_ids = np.where(index)[0] // nb_assets
    elif 1 <= nb_samples <= nb_assets:
        samples_ids = np.where(index)[0] % nb_samples
        assets_ids = np.where(index)[0] // nb_samples
    elif nb_samples > 1 == nb_assets:
        samples_ids = np.where(index)[0]
        assets_ids = np.zeros_like(samples_ids)
    else:
        samples_ids = np.zeros_like(index)
        assets_ids = np.zeros_like(samples_ids)
    return samples_ids, assets_ids


class DistributionIterator(EventIterator):
    """Concrete EventIterator for DistributionFunctions"""

    def __init__(self, model: Distribution, nb_samples, end_time, nb_assets=1):
        super().__init__(model, nb_samples, nb_assets)
        self.end_time = end_time
        self.durations = np.zeros(self.nb_assets * self.nb_samples)

    def __next__(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # shape : (nb_assets * nb_samples)
        unfailed_samples = self.durations < self.end_time
        if unfailed_samples.any():
            rvs_size = np.sum(unfailed_samples)
            event_times = self.model.rvs(size=rvs_size).reshape(-1)
            self.durations[unfailed_samples] += event_times
            samples_ids, assets_ids = transform_1d_index(
                unfailed_samples, self.nb_samples, self.nb_assets
            )
            return event_times, samples_ids, assets_ids
        else:
            raise StopIteration


class RegressionIterator(EventIterator):
    """Concrete EventIterator for RegressionFunctions"""

    def __init__(self, model: Regression, covar, nb_samples, end_time):
        super().__init__(model, nb_samples, covar.shape[0])
        self.end_time = end_time
        self.durations = np.zeros(self.nb_assets * self.nb_samples)
        self.covar = covar

    def __next__(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # shape : (nb_assets * nb_samples)
        unfailed_samples = self.durations < self.end_time
        if unfailed_samples.any():
            # shape : (nb_assets, nb_samples)
            rvs_size = self.nb_samples
            event_times = self.model.rvs(self.covar, size=rvs_size).reshape(-1)[
                unfailed_samples
            ]
            self.durations[unfailed_samples] += event_times
            samples_ids, assets_ids = transform_1d_index(
                unfailed_samples, self.nb_samples, self.nb_assets
            )
            return event_times, samples_ids, assets_ids
        else:
            raise StopIteration
