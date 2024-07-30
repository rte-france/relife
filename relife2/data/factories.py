"""
This module defines factory classes used to instanciate appropriate dataclass given user data

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from relife2.utils.types import BoolArray, FloatArray
from .dataclass import Lifetimes, ObservedLifetimes, Truncations, Deteriorations
from .tools import array_factory, lifetimes_compatibility


class LifetimeDataFactory(ABC):
    """
    Factory method of ObservedLifetimes and Truncations
    """

    def __init__(
        self,
        time: FloatArray,
        event: Optional[FloatArray] = None,
        entry: Optional[FloatArray] = None,
        departure: Optional[FloatArray] = None,
    ):

        if entry is None:
            entry = np.zeros((len(time), 1))

        if departure is None:
            departure = np.ones((len(time), 1)) * np.inf

        if event is None:
            event = np.ones((len(time), 1)).astype(np.bool_)

        self.time = time
        self.event: BoolArray = event
        self.entry: FloatArray = entry
        self.departure: FloatArray = departure

        for values in (
            self.event,
            self.entry,
            self.departure,
        ):
            if values.shape != (len(self.time), 1):
                raise ValueError("invalid argument shape")

    @abstractmethod
    def get_complete(self) -> Lifetimes:
        """
        Returns:
            Lifetimes: object containing complete lifetime values and index
        """

    @abstractmethod
    def get_left_censorships(self) -> Lifetimes:
        """
        Returns:
            Lifetimes: object containing left censorhips values and index
        """

    @abstractmethod
    def get_right_censorships(self) -> Lifetimes:
        """
        Returns:
            Lifetimes: object containing right censorhips values and index
        """

    @abstractmethod
    def get_interval_censorships(self) -> Lifetimes:
        """
        Returns:
            Lifetimes: object containing interval censorhips valuess and index
        """

    @abstractmethod
    def get_left_truncations(self) -> Lifetimes:
        """
        Returns:
            Lifetimes: object containing left truncations values and index
        """

    @abstractmethod
    def get_right_truncations(self) -> Lifetimes:
        """
        Returns:
            Lifetimes: object containing right truncations values and index
        """

    def __call__(
        self,
    ) -> tuple[
        ObservedLifetimes,
        Truncations,
    ]:
        observed_lifetimes = ObservedLifetimes(
            self.get_complete(),
            self.get_left_censorships(),
            self.get_right_censorships(),
            self.get_interval_censorships(),
        )
        truncations = Truncations(
            self.get_left_truncations(),
            self.get_right_truncations(),
        )

        try:
            lifetimes_compatibility(observed_lifetimes, truncations)
        except Exception as exc:
            raise ValueError("Incorrect input lifetimes") from exc
        return observed_lifetimes, truncations


class LifetimeDataFactoryFrom1D(LifetimeDataFactory):
    """
    Concrete implementation of LifetimeDataFactory for 1D encoding
    """

    def get_complete(self) -> Lifetimes:
        index = np.where(self.event)[0]
        values = self.time[index]
        return Lifetimes(values, index)

    def get_left_censorships(self) -> Lifetimes:
        return Lifetimes(np.empty((0, 1)), np.empty((0,), dtype=np.int64))

    def get_right_censorships(self) -> Lifetimes:
        index = np.where(~self.event)[0]
        values = self.time[index]
        return Lifetimes(values, index)

    def get_interval_censorships(self) -> Lifetimes:
        rc_index = np.where(~self.event)[0]
        rc_values = np.c_[
            self.time[rc_index], np.ones(len(rc_index)) * np.inf
        ]  # add a column of inf
        return Lifetimes(rc_values, rc_index)

    def get_left_truncations(self) -> Lifetimes:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        return Lifetimes(values, index)

    def get_right_truncations(self) -> Lifetimes:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        return Lifetimes(values, index)


class LifetimeDataFactoryFrom2D(LifetimeDataFactory):
    """
    Concrete implementation of LifetimeDataFactory for 2D encoding
    """

    def get_complete(self) -> Lifetimes:
        index = np.where(self.time[:, 0] == self.time[:, 1])[0]
        values = self.time[index, 0, None]
        return Lifetimes(values, index)

    def get_left_censorships(
        self,
    ) -> Lifetimes:
        index = np.where(self.time[:, 0] == 0)[0]
        values = self.time[index, 1, None]
        return Lifetimes(values, index)

    def get_right_censorships(
        self,
    ) -> Lifetimes:
        index = np.where(self.time[:, 1] == np.inf)[0]
        values = self.time[index, 0, None]
        return Lifetimes(values, index)

    def get_interval_censorships(self) -> Lifetimes:
        index = np.where(
            np.not_equal(self.time[:, 0], self.time[:, 1]),
        )[0]

        values = self.time[index]
        lifetimes = Lifetimes(values, index)
        if len(lifetimes) != 0:
            if np.any(lifetimes.values[:, 0] >= lifetimes.values[:, 1]):
                raise ValueError(
                    "Interval censorships lower bounds can't be higher or equal to its upper bounds"
                )
        return lifetimes

    def get_left_truncations(self) -> Lifetimes:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]  # TODO : if None, should put 0s !!
        return Lifetimes(values, index)

    def get_right_truncations(self) -> Lifetimes:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        return Lifetimes(values, index)


def lifetime_factory_template(
    time: ArrayLike,
    event: Optional[ArrayLike] = None,
    entry: Optional[ArrayLike] = None,
    departure: Optional[ArrayLike] = None,
) -> Tuple[ObservedLifetimes, Truncations]:
    """
    Args:
        time ():
        event ():
        entry ():
        departure ():

    Returns:

    """

    time = array_factory(time)

    if event is not None:
        event = array_factory(event).astype(np.bool_)

    if entry is not None:
        entry = array_factory(entry)

    if departure is not None:
        departure = array_factory(departure)

    factory: LifetimeDataFactory
    if time.shape[-1] == 1:
        factory = LifetimeDataFactoryFrom1D(
            time,
            event,
            entry,
            departure,
        )
    else:
        factory = LifetimeDataFactoryFrom2D(
            time,
            event,
            entry,
            departure,
        )
    return factory()


class DeteriorationsFactory:
    """BLABLABLA"""

    def __init__(
        self,
        deterioration_measurements: ArrayLike,
        inspection_times: ArrayLike,
        unit_ids: ArrayLike,
        initial_resistance: float,
    ):
        # verifier la cohérence des arguments (temps croissant et mesures decroissantes)
        self.deterioration_measurements = deterioration_measurements
        self.inspection_times = inspection_times
        self.unit_ids = unit_ids
        self.initial_resistance = initial_resistance

    def __call__(self) -> Deteriorations:
        sorted_indices = np.argsort(self.unit_ids, kind="mergesort")
        sorted_unit_ids = self.unit_ids[sorted_indices]
        sorted_deteriorations_measurements = self.deterioration_measurements[
            sorted_indices
        ]
        sorted_inspection_times = self.inspection_times[sorted_indices]
        unique_unit_ids, counts = np.unique(sorted_unit_ids, return_counts=True)

        max_len = np.max(counts)
        split_indices = np.cumsum(counts)[:-1]

        deteriorations_measurements_2d = np.vstack(
            [
                np.concatenate((split_arr, np.ones(max_len - len(split_arr)) * np.nan))
                for split_arr in np.split(
                    sorted_deteriorations_measurements, split_indices
                )
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
