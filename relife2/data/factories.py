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

from relife2.types import BoolArray, FloatArray
from .dataclass import Lifetimes, ObservedLifetimes, Truncations, Deteriorations
from .tools import array_factory, lifetimes_compatibility


class LifetimeDataFactory(ABC):
    """
    Factory method of ObservedLifetimes and Truncations
    """

    def __init__(
        self,
        time: FloatArray,
        entry: Optional[FloatArray] = None,
        departure: Optional[FloatArray] = None,
        lc_indicators: Optional[BoolArray] = None,
        rc_indicators: Optional[BoolArray] = None,
    ):

        if entry is None:
            entry = np.zeros((len(time), 1))

        if departure is None:
            departure = np.ones((len(time), 1)) * np.inf

        if lc_indicators is None:
            lc_indicators = np.zeros((len(time), 1)).astype(np.bool_)

        if rc_indicators is None:
            rc_indicators = np.zeros((len(time), 1)).astype(np.bool_)

        self.time = time
        self.entry: FloatArray = entry
        self.departure: FloatArray = departure
        self.lc_indicators: BoolArray = lc_indicators
        self.rc_indicators: BoolArray = rc_indicators

        if np.any(np.logical_and(self.lc_indicators, self.rc_indicators)) is True:
            raise ValueError(
                """
                lc_indicators and rc_indicators can't be true at the same index
                """
            )

        for values in (
            self.entry,
            self.departure,
            self.lc_indicators,
            self.rc_indicators,
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
        except Exception as error:
            raise ValueError("Incorrect input lifetimes") from error
        return observed_lifetimes, truncations


class LifetimeDataFactoryFrom1D(LifetimeDataFactory):
    """
    Concrete implementation of LifetimeDataFactory for 1D encoding
    """

    def get_complete(self) -> Lifetimes:
        index = np.where(
            np.logical_and(np.invert(self.lc_indicators), np.invert(self.rc_indicators))
        )[0]
        values = self.time[index]
        return Lifetimes(values, index)

    def get_left_censorships(self) -> Lifetimes:
        index = np.where(self.lc_indicators)[0]
        values = self.time[index]
        return Lifetimes(values, index)

    def get_right_censorships(self) -> Lifetimes:
        index = np.where(self.rc_indicators)[0]
        values = self.time[index]
        return Lifetimes(values, index)

    def get_interval_censorships(self) -> Lifetimes:
        return Lifetimes(np.empty((0, 2)), np.empty((0,), dtype=np.int64))

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
        index = np.where(self.time[:, 0] == 0.0)[0]
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
            np.logical_and(
                np.logical_and(
                    self.time[:, 0] > 0,
                    self.time[:, 1] < np.inf,
                ),
                np.not_equal(self.time[:, 0], self.time[:, 1]),
            )
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
        values = self.entry[index]
        return Lifetimes(values, index)

    def get_right_truncations(self) -> Lifetimes:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        return Lifetimes(values, index)


def lifetime_factory_template(
    time: ArrayLike,
    entry: Optional[ArrayLike] = None,
    departure: Optional[ArrayLike] = None,
    lc_indicators: Optional[ArrayLike] = None,
    rc_indicators: Optional[ArrayLike] = None,
) -> Tuple[ObservedLifetimes, Truncations]:
    """
    Args:
        time ():
        entry ():
        departure ():
        lc_indicators ():
        rc_indicators ():

    Returns:

    """

    time = array_factory(time)

    if entry is not None:
        entry = array_factory(entry)

    if departure is not None:
        departure = array_factory(departure)

    if lc_indicators is not None:
        lc_indicators = array_factory(lc_indicators).astype(np.bool_)

    if rc_indicators is not None:
        rc_indicators = array_factory(rc_indicators).astype(np.bool_)

    factory: LifetimeDataFactory
    if time.shape[-1] == 1:
        factory = LifetimeDataFactoryFrom1D(
            time,
            entry,
            departure,
            lc_indicators,
            rc_indicators,
        )
    else:
        factory = LifetimeDataFactoryFrom2D(
            time,
            entry,
            departure,
            lc_indicators,
            rc_indicators,
        )
    return factory()


class DeteriorationsFactory:
    """BLABLABLA"""

    def __init__(
        self,
        deterioration_measurements: ArrayLike,
        inspection_times: ArrayLike,
        unit_ids: ArrayLike,
    ):
        # verifier la cohérence des arguments
        self.deterioration_measurements = deterioration_measurements
        self.inspection_times = inspection_times
        self.unit_ids = unit_ids

    @staticmethod
    def _padding_values(max_length, arrays):
        return np.array(
            list(
                map(
                    lambda arr: np.concatenate(
                        [arr, np.full(max_length - len(arr), np.nan)]
                    ),
                    arrays,
                )
            )
        )

    def get_data(self) -> FloatArray:
        sorted_indices = np.argsort(self.unit_ids)
        sorted_ids = self.unit_ids[sorted_indices]
        sorted_inspection_times = self.inspection_times[sorted_indices]
        arrays_inspection_times = np.split(
            sorted_inspection_times, np.unique(sorted_ids, return_index=True)[1]
        )[1:]
        max_len = max(len(arr) for arr in arrays_inspection_times)
        inspection_times = DeteriorationsFactory._padding_values(
            max_len, arrays_inspection_times
        )
        sorted_deterioration_measurements = self.deterioration_measurements[
            sorted_indices
        ]
        arrays_deterioration_measurements = np.split(
            sorted_deterioration_measurements,
            np.unique(sorted_ids, return_index=True)[1],
        )[1:]
        deterioration_measurements = DeteriorationsFactory._padding_values(
            max_len, arrays_deterioration_measurements
        )
        return inspection_times, deterioration_measurements

    def fabric(self) -> Deteriorations:
        """BLABLABLA"""
        deterioration_measurements, inspection_times = self.get_data()
        return Deteriorations(
            deterioration_measurements,
            inspection_times,
            self.unit_ids,
        )
