"""
This module defines dataclass used to encapsulate data used for parameters estimation of models

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Iterator, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class IndexedData:
    """
    Object that encapsulates lifetime data values and corresponding units index
    """

    values: NDArray[np.float64]
    index: NDArray[np.int64]  # arrays of int

    def __post_init__(self):
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)
        if self.index.ndim != 1:
            raise ValueError("Invalid LifetimeData unit_ids number of dimensions")
        if len(self.values) != len(self.index):
            raise ValueError("Incompatible lifetime values and unit_ids")

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class LifetimeData:
    """BLABLABLA"""

    complete: IndexedData
    left_censored: IndexedData
    right_censored: IndexedData
    interval_censored: IndexedData

    def __post_init__(self):
        self.rc = IndexedData(
            np.concatenate(
                (
                    self.complete.values,
                    self.right_censored.values,
                ),
                axis=0,
            ),
            np.concatenate((self.complete.index, self.right_censored.index)),
        )
        self.rlc = IndexedData(
            np.concatenate(
                [
                    self.complete.values,
                    self.left_censored.values,
                    self.right_censored.values,
                ]
            ),
            np.concatenate(
                (
                    self.complete.index,
                    self.left_censored.index,
                    self.right_censored.index,
                )
            ),
        )


def intersect_lifetimes(*lifetimes: IndexedData) -> list[IndexedData]:
    """
    Args:
        *lifetimes: LifetimeData object.s containing values of shape (n1, p1), (n2, p2), etc.

    Returns:

    Examples:
        >>> lifetimes_1 = IndexedData(values = np.array([[1.], [2.]]), index = np.array([3, 10]))
        >>> lifetimes_2 = IndexedData(values = np.array([[3.], [5.]]), index = np.array([10, 2]))
        >>> intersect_lifetimes(lifetimes_1, lifetimes_2)
        [Lifetimes(values=array([[2]]), index=array([10])), Lifetimes(values=array([[3]]), index=array([10]))]
    """

    inter_ids = np.array(
        list(set.intersection(*[set(_lifetimes.index) for _lifetimes in lifetimes]))
    ).astype(np.int64)
    return [
        IndexedData(
            _lifetimes.values[np.isin(_lifetimes.index, inter_ids)],
            inter_ids,
        )
        for _lifetimes in lifetimes
    ]


@dataclass
class Truncations:
    """BLABLABLA"""

    left: IndexedData
    right: IndexedData


@dataclass
class Deteriorations:
    """BLABLABLA"""

    values: np.ndarray  # R0 in first column (always)
    times: np.ndarray  # 0 in first column (always)
    ids: np.ndarray

    def __post_init__(self):
        # self.values = np.ma.array(self.values, mask=np.isnan(self.values))
        # self.times = np.ma.array(self.times, mask=np.isnan(self.times))
        self.increments = -np.diff(self.values, axis=1)
        self.event = self.increments == 0


def lifetimes_compatibility(
    observed_lifetimes: LifetimeData, truncations: Truncations
) -> None:
    """
    Check the compatibility between each observed lifetimes and truncation values
    Args:
        observed_lifetimes ():
        truncations ():
    """

    for attr_name in [
        "complete",
        "left_censored",
        "right_censored",
        "interval_censored",
    ]:
        lifetimes = getattr(observed_lifetimes, attr_name)
        if len(truncations.left) != 0 and len(lifetimes) != 0:
            left_truncated_lifetimes = intersect_lifetimes(lifetimes, truncations.left)
            if len(left_truncated_lifetimes) != 0:
                if np.any(
                    np.min(
                        np.where(
                            left_truncated_lifetimes[0].values == 0,
                            left_truncated_lifetimes[1].values,
                            left_truncated_lifetimes[0].values,
                        ),
                        axis=1,
                        keepdims=True,
                    )
                    < left_truncated_lifetimes[1].values
                ):
                    raise ValueError("Some lifetimes are under left truncation bounds")
        if len(truncations.right) != 0 and len(lifetimes) != 0:
            right_truncated_lifetimes = intersect_lifetimes(
                lifetimes, truncations.right
            )
            if len(right_truncated_lifetimes) != 0:
                if np.any(
                    np.max(
                        np.where(
                            right_truncated_lifetimes[0].values == np.inf,
                            right_truncated_lifetimes[1].values,
                            right_truncated_lifetimes[0].values,
                        ),
                        axis=1,
                        keepdims=True,
                    )
                    > right_truncated_lifetimes[1].values
                ):
                    raise ValueError("Some lifetimes are above right truncation bounds")


class LifetimesFactory(ABC):
    """
    Factory method of ObservedLifetimes and Truncations
    """

    def __init__(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ):

        if entry is None:
            entry = np.zeros((len(time), 1))

        if departure is None:
            departure = np.ones((len(time), 1)) * np.inf

        if event is None:
            event = np.ones((len(time), 1)).astype(np.bool_)

        self.time = time
        self.event: NDArray[np.bool_] = event.astype(np.bool_)
        self.entry: NDArray[np.float64] = entry
        self.departure: NDArray[np.float64] = departure

    @abstractmethod
    def get_complete(self) -> IndexedData:
        """
        Returns:
            IndexedData: object containing complete lifetime values and index
        """

    @abstractmethod
    def get_left_censorships(self) -> IndexedData:
        """
        Returns:
            IndexedData: object containing left censorhips values and index
        """

    @abstractmethod
    def get_right_censorships(self) -> IndexedData:
        """
        Returns:
            IndexedData: object containing right censorhips values and index
        """

    @abstractmethod
    def get_interval_censorships(self) -> IndexedData:
        """
        Returns:
            IndexedData: object containing interval censorhips valuess and index
        """

    @abstractmethod
    def get_left_truncations(self) -> IndexedData:
        """
        Returns:
            IndexedData: object containing left truncations values and index
        """

    @abstractmethod
    def get_right_truncations(self) -> IndexedData:
        """
        Returns:
            IndexedData: object containing right truncations values and index
        """

    def __call__(
        self,
    ) -> tuple[
        LifetimeData,
        Truncations,
    ]:
        observed_lifetimes = LifetimeData(
            self.get_complete(),
            self.get_left_censorships(),
            self.get_right_censorships(),
            self.get_interval_censorships(),
        )
        truncations = Truncations(
            self.get_left_truncations(), self.get_right_truncations()
        )

        try:
            lifetimes_compatibility(observed_lifetimes, truncations)
        except Exception as exc:
            raise ValueError("Incorrect input lifetimes") from exc
        return observed_lifetimes, truncations


class LifetimeDataFactoryFrom1D(LifetimesFactory):
    """
    Concrete implementation of LifetimeDataFactory for 1D encoding
    """

    def get_complete(self) -> IndexedData:
        index = np.where(self.event)[0]
        values = self.time[index]
        return IndexedData(values, index)

    def get_left_censorships(self) -> IndexedData:
        return IndexedData(
            np.empty((0, 1), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )

    def get_right_censorships(self) -> IndexedData:
        index = np.where(~self.event)[0]
        values = self.time[index]
        return IndexedData(values, index)

    def get_interval_censorships(self) -> IndexedData:
        rc_index = np.where(~self.event)[0]
        rc_values = np.c_[
            self.time[rc_index], np.ones(len(rc_index)) * np.inf
        ]  # add a column of inf
        return IndexedData(rc_values, rc_index)

    def get_left_truncations(self) -> IndexedData:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        return IndexedData(values, index)

    def get_right_truncations(self) -> IndexedData:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        return IndexedData(values, index)


class LifetimeDataFactoryFrom2D(LifetimesFactory):
    """
    Concrete implementation of LifetimeDataFactory for 2D encoding
    """

    def get_complete(self) -> IndexedData:
        index = np.where(self.time[:, 0] == self.time[:, 1])[0]
        values = self.time[index, 0, None]
        return IndexedData(values, index)

    def get_left_censorships(
        self,
    ) -> IndexedData:
        index = np.where(self.time[:, 0] == 0)[0]
        values = self.time[index, 1, None]
        return IndexedData(values, index)

    def get_right_censorships(
        self,
    ) -> IndexedData:
        index = np.where(self.time[:, 1] == np.inf)[0]
        values = self.time[index, 0, None]
        return IndexedData(values, index)

    def get_interval_censorships(self) -> IndexedData:
        index = np.where(
            np.not_equal(self.time[:, 0], self.time[:, 1]),
        )[0]

        values = self.time[index]
        lifetimes = IndexedData(values, index)
        if len(lifetimes) != 0:
            if np.any(lifetimes.values[:, 0] >= lifetimes.values[:, 1]):
                raise ValueError(
                    "Interval censorships lower bounds can't be higher or equal to its upper bounds"
                )
        return lifetimes

    def get_left_truncations(self) -> IndexedData:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        return IndexedData(values, index)

    def get_right_truncations(self) -> IndexedData:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        return IndexedData(values, index)


def lifetime_factory_template(
    time: NDArray[np.float64],
    event: Optional[NDArray[np.bool_]] = None,
    entry: Optional[NDArray[np.float64]] = None,
    departure: Optional[NDArray[np.float64]] = None,
) -> tuple[LifetimeData, Truncations]:
    """
    Args:
        time ():
        event ():
        entry ():
        departure ():

    Returns:

    """
    factory: LifetimesFactory
    if time.ndim == 1:
        factory = LifetimeDataFactoryFrom1D(time, event, entry, departure)
    elif time.ndim == 2:
        if time.shape[-1] != 2:
            raise ValueError("If time ndim is 2, time shape must be (n, 2)")
        factory = LifetimeDataFactoryFrom2D(time, event, entry, departure)
    else:
        raise ValueError("time ndim must be 1 or 2")
    return factory()


def deteriorations_factory(
    deterioration_measurements: np.ndarray,
    inspection_times: np.ndarray,
    unit_ids: np.ndarray,
):
    """
    Args:
        deterioration_measurements ():
        inspection_times ():
        unit_ids ():
    Returns:

    """
    # verifier la cohérence des arguments (temps croissant et mesures decroissantes)

    sorted_indices = np.argsort(unit_ids, kind="mergesort")
    sorted_unit_ids = unit_ids[sorted_indices]
    sorted_deteriorations_measurements = deterioration_measurements[sorted_indices]
    sorted_inspection_times = inspection_times[sorted_indices]
    unique_unit_ids, counts = np.unique(sorted_unit_ids, return_counts=True)

    max_len = np.max(counts)
    split_indices = np.cumsum(counts)[:-1]

    deteriorations_measurements_2d = np.vstack(
        [
            np.concatenate((split_arr, np.ones(max_len - len(split_arr)) * np.nan))
            for split_arr in np.split(sorted_deteriorations_measurements, split_indices)
        ]
    )

    inspection_times_2d = np.vstack(
        [
            np.concatenate((split_arr, np.ones(max_len - len(split_arr)) * np.nan))
            for split_arr in np.split(sorted_inspection_times, split_indices)
        ]
    )

    return Deteriorations(
        deteriorations_measurements_2d,
        inspection_times_2d,
        unique_unit_ids,
    )


@dataclass
class CountData:
    samples: NDArray[np.int64] = field(repr=False)  # samples index
    assets: NDArray[np.int64] = field(repr=False)  # assets index
    order: NDArray[np.int64] = field(
        repr=False
    )  # order index (order in generation process)
    event_times: NDArray[np.float64] = field(repr=False)

    nb_samples: int = field(init=False)
    nb_assets: int = field(init=False)
    samples_index: NDArray[np.int64] = field(
        init=False, repr=False
    )  # unique samples index
    assets_index: NDArray[np.int64] = field(
        init=False, repr=False
    )  # unique assets index

    def __post_init__(self):
        fields_values = [
            getattr(self, _field.name) for _field in fields(self) if _field.init
        ]
        if not all(arr.ndim == 1 for arr in fields_values):
            raise ValueError("All array values must be 1d")
        if not len(set(arr.shape[0] for arr in fields_values)) == 1:
            raise ValueError("All array values must have the same shape")

        self.samples_index = np.unique(self.samples)  # samples unique index
        self.assets_index = np.unique(self.assets)  # assets unique index
        self.nb_samples = len(self.samples_index)
        self.nb_assets = len(self.assets_index)

    def __len__(self) -> int:
        return self.nb_samples * self.nb_assets

    def number_of_events(
        self, sample: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ind = self.samples == sample
        times = np.insert(np.sort(self.event_times[ind]), 0, 0)
        counts = np.arange(times.size)
        return times, counts

    def mean_number_of_events(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        times = np.insert(np.sort(self.event_times), 0, 0)
        counts = np.arange(times.size) / self.nb_samples
        return times, counts

    def iter(self):
        return CountDataIterable(self)


class CountDataIterable:
    def __init__(self, data: CountData):
        self.data = data

        sorted_index = np.lexsort(
            (self.data.order, self.data.assets, self.data.samples)
        )
        self.sorted_fields = {
            _field.name: getattr(self.data, _field.name).copy()[sorted_index]
            for _field in fields(self.data)
            if _field.init
        }

    def __len__(self) -> int:
        return self.data.nb_samples * self.data.nb_assets

    def __iter__(self) -> Iterator[tuple[int, int, dict[str, NDArray[np.float64]]]]:

        for sample in self.data.samples_index:
            sample_mask = self.sorted_fields["samples"] == sample
            for asset in self.data.assets_index:
                asset_mask = self.sorted_fields["assets"][sample_mask] == asset
                values_dict = {
                    k: v[sample_mask][asset_mask]
                    for k, v in self.sorted_fields.items()
                    if k not in ("samples", "assets", "events")
                }
                yield int(sample), int(asset), values_dict


@dataclass
class RenewalData(CountData):
    lifetimes: NDArray[np.float64] = field(repr=False)
    events: NDArray[np.bool_] = field(
        repr=False
    )  # event indicators (right censored or not)

    # TODO: remove model, model_args from there. Not needed
    # 1. if init_params uses *args (see model.py) then
    # 2. init_params in Regression does not rely on covar stored in Sample
    # 3. Sample does not need to store args
    # 4. to_lifetime_data only returns a object constructing on time, event, entry only
    def to_lifetime_data(
        self,
        t0: float = 0,
        tf: Optional[float] = None,
        sample: Optional[int] = None,
    ) -> LifetimeData:
        if t0 >= tf:
            raise ValueError("`t0` must be strictly lesser than `tf`")

        # Filtering sample and sorting by times
        s = self.samples == sample if sample is not None else Ellipsis
        order = np.argsort(self.event_times[s])
        indices = self.assets[s][order]
        samples = self.samples[s][order]
        uindices = np.ravel_multi_index(
            (indices, samples), (self.nb_assets, self.nb_samples)
        )
        event_times = self.event_times[s][order]
        lifetimes = self.lifetimes[s][order]
        events = self.events[s][order]

        # Indices of interest
        ind0 = (event_times > t0) & (
            event_times <= tf
        )  # Indices of replacement occuring inside the obervation window
        ind1 = (
            event_times > tf
        )  # Indices of replacement occuring after the observation window which include right censoring

        # Replacements occuring inside the observation window
        time0 = lifetimes[ind0]
        event0 = events[ind0]
        entry0 = np.zeros(time0.size)
        _, lt = np.unique(
            uindices[ind0], return_index=True
        )  # get the indices of the first replacements ocurring in the observation window
        b0 = (
            event_times[ind0][lt] - lifetimes[ind0][lt]
        )  # time at birth for the firt replacements
        entry0[lt] = np.where(b0 >= t0, 0, t0 - b0)

        # Right censoring
        _, rc = np.unique(uindices[ind1], return_index=True)
        bf = (
            event_times[ind1][rc] - lifetimes[ind1][rc]
        )  # time at birth for the right censored
        b1 = bf[
            bf < tf
        ]  # ensure that time of birth for the right censored is not equal to tf.
        time1 = tf - b1
        event1 = np.zeros(b1.size)
        entry1 = np.where(b1 >= t0, 0, t0 - b1)

        # Concatenate
        time = np.concatenate((time0, time1))
        event = np.concatenate((event0, event1))
        entry = np.concatenate((entry0, entry1))

        return lifetime_factory_template(time, event, entry)


@dataclass
class RenewalRewardData(RenewalData):
    total_rewards: NDArray[np.float64] = field(repr=False)

    def cum_total_rewards(
        self, sample: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ind = self.samples == sample
        s = np.argsort(self.event_times[ind])
        times = np.insert(self.event_times[ind][s], 0, 0)
        z = np.insert(self.total_rewards[ind][s].cumsum(), 0, 0)
        return times, z

    def mean_total_reward(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        s = np.argsort(self.event_times)
        times = np.insert(self.event_times[s], 0, 0)
        z = np.insert(self.total_rewards[s].cumsum(), 0, 0) / self.nb_samples
        return times, z
