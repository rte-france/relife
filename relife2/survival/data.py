"""
This module gathers class used by subsystems' objects to treat lifetime data

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def array_factory(obj: ArrayLike) -> FloatArray:
    """
    Converts object input to 2d array of shape (n, p)
    n is the number of units
    p is the number of points
    Args:
        obj: object input

    Returns:
        FloatArray: 2d array
    """
    try:
        obj = np.asarray(obj, dtype=np.float64)
    except Exception as error:
        raise ValueError("Invalid type of param") from error
    if obj.ndim < 2:
        obj = obj.reshape(-1, 1)
    elif obj.ndim > 2:
        raise ValueError(
            f"input FloatArray can't have more than 2 dimensions : got {obj}"
        )
    return obj


@dataclass(frozen=True)
class LifetimeData:
    """
    Object that encapsulates lifetime data values and corresponding units index
    """

    values: FloatArray
    unit_ids: IntArray

    def __post_init__(self):
        if self.values.ndim != 2:
            raise ValueError("Invalid LifetimeData values number of dimensions")
        if self.unit_ids.ndim != 1:
            raise ValueError("Invalid LifetimeData unit_ids number of dimensions")
        if len(self.values) != len(self.unit_ids):
            raise ValueError("Incompatible Measures values and unit_ids")

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class ObservedLifetimes:
    """BLABLABLA"""

    complete: LifetimeData
    left_censored: LifetimeData
    right_censored: LifetimeData
    interval_censored: LifetimeData


@dataclass(frozen=True)
class Truncations:
    """BLABLABLA"""

    left: LifetimeData
    right: LifetimeData


def intersect_lifetime_data(*lifetime_data: LifetimeData) -> LifetimeData:
    """
    Args:
        *lifetime_data: Measures object.s containing values of shape (n1, p1), (n2, p2), etc.

    Returns:
        LifetimeData: One Measures object where values are concatanation of common units values. The result
        is of shape (N, p1 + p2 + ...).

    Examples:
        >>> lifetime_data_1 = LifetimeData(values = np.array([[1], [2]]), unit_ids = np.array([3, 10]))
        >>> lifetime_data_2 = LifetimeData(values = np.array([[3], [5]]), unit_ids = np.array([10, 2]))
        >>> intersect_lifetime_data(lifetime_data_1, lifetime_data_2)
        LifetimeData(values=array([[2, 3]]), unit_ids=array([10]))
    """

    inter_ids = np.array(
        list(set.intersection(*[set(m.unit_ids) for m in lifetime_data]))
    )
    return LifetimeData(
        np.hstack([m.values[np.isin(m.unit_ids, inter_ids)] for m in lifetime_data]),
        inter_ids,
    )


class LifetimeDataFactory(ABC):
    """
    Factory method of Measures object
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
            lc_indicators = np.zeros_like(time).astype(np.bool_)

        if rc_indicators is None:
            rc_indicators = np.zeros_like(time).astype(np.bool_)

        self.time = time
        self.entry = entry
        self.departure = departure
        self.lc_indicators = lc_indicators
        self.rc_indicators = rc_indicators
        self._check_format()

        if np.any(np.logical_and(self.lc_indicators, self.rc_indicators)) is True:
            raise ValueError(
                """
                lc_indicators and rc_indicators can't be true at the same index
                """
            )

    def _check_format(self) -> None:
        for values in (
            self.entry,
            self.departure,
            self.lc_indicators,
            self.rc_indicators,
        ):
            if values.shape != (len(self.time), 1):
                raise ValueError("invalid argument shape")

    @abstractmethod
    def get_complete(self) -> LifetimeData:
        """
        Returns:
            LifetimeData: object containing complete lifetime values and index
        """

    @abstractmethod
    def get_left_censorships(self) -> LifetimeData:
        """
        Returns:
            LifetimeData: object containing left censorhips values and index
        """

    @abstractmethod
    def get_right_censorships(self) -> LifetimeData:
        """
        Returns:
            LifetimeData: object containing right censorhips values and index
        """

    @abstractmethod
    def get_interval_censorships(self) -> LifetimeData:
        """
        Returns:
            LifetimeData: object containing interval censorhips valuess and index
        """

    @abstractmethod
    def get_left_truncations(self) -> LifetimeData:
        """
        Returns:
            LifetimeData: object containing left truncations values and index
        """

    @abstractmethod
    def get_right_truncations(self) -> LifetimeData:
        """
        Returns:
            LifetimeData: object containing right truncations values and index
        """

    @staticmethod
    def _compatible_with_left_truncations(
        lifetimes: LifetimeData, left_truncations: LifetimeData
    ) -> None:
        if len(lifetimes) != 0 and len(left_truncations) != 0:
            intersected_measures = intersect_lifetime_data(lifetimes, left_truncations)
            if len(intersected_measures) != 0:
                if np.any(
                    np.min(
                        intersected_measures.values[:, : lifetimes.values.shape[-1]],
                        axis=1,
                        keepdims=True,
                    )
                    < intersected_measures.values[:, lifetimes.values.shape[-1] :]
                ):
                    raise ValueError(
                        f"""
                        Some lifetimes are under left truncation bounds :
                        {lifetimes} and {left_truncations}
                        """
                    )

    @staticmethod
    def _compatible_with_right_truncations(
        lifetimes: LifetimeData, right_truncations: LifetimeData
    ) -> None:
        if len(lifetimes) != 0 and len(right_truncations) != 0:
            intersected_measures = intersect_lifetime_data(lifetimes, right_truncations)
            if len(intersected_measures) != 0:
                if np.any(
                    np.max(
                        intersected_measures.values[:, : lifetimes.values.shape[-1]],
                        axis=1,
                        keepdims=True,
                    )
                    > intersected_measures.values[:, lifetimes.values.shape[-1] :]
                ):
                    raise ValueError(
                        f"""
                        Some lifetimes are above right truncation bounds :
                        {lifetimes} and {right_truncations}
                        """
                    )

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
            for lifetimes in observed_lifetimes.__annotations__.values():
                LifetimeDataFactory._compatible_with_left_truncations(
                    lifetimes, truncations.left
                )
                LifetimeDataFactory._compatible_with_right_truncations(
                    lifetimes, truncations.right
                )
        except Exception as error:
            raise ValueError("Incorrect input lifetimes") from error
        return observed_lifetimes, truncations


class LifetimeDataFactoryFrom1D(LifetimeDataFactory):
    """
    Concrete implementation of MeasuresFactory for 1D encoding
    """

    def get_complete(self) -> LifetimeData:
        index = np.where(np.logical_and(~self.lc_indicators, ~self.rc_indicators))[0]
        values = self.time[index]
        return LifetimeData(values, index)

    def get_left_censorships(self) -> LifetimeData:
        index = np.where(self.lc_indicators)[0]
        values = self.time[index]
        return LifetimeData(values, index)

    def get_right_censorships(self) -> LifetimeData:
        index = np.where(self.rc_indicators)[0]
        values = self.time[index]
        return LifetimeData(values, index)

    def get_interval_censorships(self) -> LifetimeData:
        return LifetimeData(np.empty((0, 2)), np.empty((0,), dtype=np.int64))

    def get_left_truncations(self) -> LifetimeData:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        return LifetimeData(values, index)

    def get_right_truncations(self) -> LifetimeData:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        return LifetimeData(values, index)


class LifetimeDataFactoryFrom2D(LifetimeDataFactory):
    """
    Concrete implementation of MeasuresFactory for 2D encoding
    """

    def get_complete(self) -> LifetimeData:
        index = np.where(self.time[:, 0] == self.time[:, 1])[0]
        values = self.time[index, 0, None]
        return LifetimeData(values, index)

    def get_left_censorships(
        self,
    ) -> LifetimeData:
        index = np.where(self.time[:, 0] == 0.0)[0]
        values = self.time[index, 1, None]
        return LifetimeData(values, index)

    def get_right_censorships(
        self,
    ) -> LifetimeData:
        index = np.where(self.time[:, 1] == np.inf)[0]
        values = self.time[index, 0, None]
        return LifetimeData(values, index)

    def get_interval_censorships(self) -> LifetimeData:
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
        measures = LifetimeData(values, index)
        if len(measures) != 0:
            if np.any(measures.values[:, 0] >= measures.values[:, 1]):
                raise ValueError(
                    "Interval censorships lower bounds can't be higher or equal to its upper bounds"
                )
        return measures

    def get_left_truncations(self) -> LifetimeData:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        return LifetimeData(values, index)

    def get_right_truncations(self) -> LifetimeData:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        return LifetimeData(values, index)
