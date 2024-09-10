"""
This module defines factory classes used to instanciate appropriate dataclass given user data

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .dataclass import (
    Deteriorations,
    LifetimeSample,
    Sample,
    Truncations,
    intersect_lifetimes,
)


def lifetimes_compatibility(
    observed_lifetimes: LifetimeSample, truncations: Truncations
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
        args: tuple[NDArray[np.float64], ...] | tuple[()] = (),
    ):

        if entry is None:
            entry = np.zeros((len(time), 1))

        if departure is None:
            departure = np.ones((len(time), 1)) * np.inf

        if event is None:
            event = np.ones((len(time), 1)).astype(np.bool_)

        self.time = time
        self.event: np.ndarray = event.astype(np.bool_)
        self.entry: np.ndarray = entry
        self.departure: np.ndarray = departure
        self.args = args

    @abstractmethod
    def get_complete(self) -> Sample:
        """
        Returns:
            Sample: object containing complete lifetime values and index
        """

    @abstractmethod
    def get_left_censorships(self) -> Sample:
        """
        Returns:
            Sample: object containing left censorhips values and index
        """

    @abstractmethod
    def get_right_censorships(self) -> Sample:
        """
        Returns:
            Sample: object containing right censorhips values and index
        """

    @abstractmethod
    def get_interval_censorships(self) -> Sample:
        """
        Returns:
            Sample: object containing interval censorhips valuess and index
        """

    @abstractmethod
    def get_left_truncations(self) -> Sample:
        """
        Returns:
            Sample: object containing left truncations values and index
        """

    @abstractmethod
    def get_right_truncations(self) -> Sample:
        """
        Returns:
            Sample: object containing right truncations values and index
        """

    def __call__(
        self,
    ) -> tuple[
        LifetimeSample,
        Truncations,
    ]:
        observed_lifetimes = LifetimeSample(
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

    def get_complete(self) -> Sample:
        index = np.where(self.event)[0]
        values = self.time[index]
        args = [v[index] for v in self.args]
        return Sample(values, index, args)

    def get_left_censorships(self) -> Sample:
        return Sample(
            np.empty((0, 1), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            [np.empty((0, v.shape[-1]), dtype=np.float64) for v in self.args],
        )

    def get_right_censorships(self) -> Sample:
        index = np.where(~self.event)[0]
        values = self.time[index]
        args = [v[index] for v in self.args]
        return Sample(values, index, args)

    def get_interval_censorships(self) -> Sample:
        rc_index = np.where(~self.event)[0]
        rc_values = np.c_[
            self.time[rc_index], np.ones(len(rc_index)) * np.inf
        ]  # add a column of inf
        args = [v[rc_index] for v in self.args]
        return Sample(rc_values, rc_index, args)

    def get_left_truncations(self) -> Sample:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        args = [v[index] for v in self.args]
        return Sample(values, index, args)

    def get_right_truncations(self) -> Sample:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        args = [v[index] for v in self.args]
        return Sample(values, index, args)


class LifetimeDataFactoryFrom2D(LifetimesFactory):
    """
    Concrete implementation of LifetimeDataFactory for 2D encoding
    """

    def get_complete(self) -> Sample:
        index = np.where(self.time[:, 0] == self.time[:, 1])[0]
        values = self.time[index, 0, None]
        args = [v[index] for v in self.args]
        return Sample(values, index, args)

    def get_left_censorships(
        self,
    ) -> Sample:
        index = np.where(self.time[:, 0] == 0)[0]
        values = self.time[index, 1, None]
        args = [v[index] for v in self.args]
        return Sample(values, index, args)

    def get_right_censorships(
        self,
    ) -> Sample:
        index = np.where(self.time[:, 1] == np.inf)[0]
        values = self.time[index, 0, None]
        args = [v[index] for v in self.args]
        return Sample(values, index, args)

    def get_interval_censorships(self) -> Sample:
        index = np.where(
            np.not_equal(self.time[:, 0], self.time[:, 1]),
        )[0]

        values = self.time[index]
        args = [v[index] for v in self.args]
        lifetimes = Sample(values, index, args)
        if len(lifetimes) != 0:
            if np.any(lifetimes.values[:, 0] >= lifetimes.values[:, 1]):
                raise ValueError(
                    "Interval censorships lower bounds can't be higher or equal to its upper bounds"
                )
        return lifetimes

    def get_left_truncations(self) -> Sample:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        args = [v[index] for v in self.args]
        return Sample(values, index, args)

    def get_right_truncations(self) -> Sample:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        args = [v[index] for v in self.args]
        return Sample(values, index, args)


def lifetime_factory_template(
    time: NDArray[np.float64],
    event: Optional[NDArray[np.bool_]] = None,
    entry: Optional[NDArray[np.float64]] = None,
    departure: Optional[NDArray[np.float64]] = None,
    args: tuple[NDArray[np.float64], ...] | tuple[()] = (),
) -> Tuple[LifetimeSample, Truncations]:
    """
    Args:
        time ():
        event ():
        entry ():
        departure ():
        args ():

    Returns:

    """
    factory: LifetimesFactory
    if time.ndim == 1:
        factory = LifetimeDataFactoryFrom1D(time, event, entry, departure, args)
    elif time.ndim == 2:
        if time.shape[-1] != 2:
            raise ValueError("If time ndim is 2, time shape must be (n, 2)")
        factory = LifetimeDataFactoryFrom2D(time, event, entry, departure, args)
    else:
        raise ValueError("time ndim must be 1 or 2")
    return factory()


def deteriorations_factory(
    deterioration_measurements: np.ndarray,
    inspection_times: np.ndarray,
    unit_ids: np.ndarray,
    initial_resistance: float,
):
    """
    Args:
        deterioration_measurements ():
        inspection_times ():
        unit_ids ():
        initial_resistance ():
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
