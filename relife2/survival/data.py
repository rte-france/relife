"""
This module gathers class used by subsystems' objects to treat lifetime data

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

# pylint: disable=invalid-unary-operand-type


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
            raise ValueError("Incompatible lifetime values and unit_ids")

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class ObservedLifetimes:
    """BLABLABLA"""

    complete: LifetimeData
    left_censored: LifetimeData
    right_censored: LifetimeData
    interval_censored: LifetimeData

    def __post_init__(self):
        self.rc = LifetimeData(
            np.concatenate(
                (
                    self.complete.values,
                    self.right_censored.values,
                ),
                axis=0,
            ),
            np.concatenate(self.complete.unit_ids, self.right_censored.unit_ids),
        )
        self.rlc = LifetimeData(
            np.concatenate(
                [
                    self.complete.values,
                    self.left_censored.values,
                    self.right_censored.values,
                ]
            ),
            np.concatenate(
                self.complete.unit_ids,
                self.left_censored.unit_ids,
                self.right_censored.unit_ids,
            ),
        )


@dataclass(frozen=True)
class Truncations:
    """BLABLABLA"""

    left: LifetimeData
    right: LifetimeData


def intersect_lifetimes(*lifetimes: LifetimeData) -> LifetimeData:
    """
    Args:
        *lifetimes: LifetimeData object.s containing values of shape (n1, p1), (n2, p2), etc.

    Returns:
        LifetimeData: One LifetimeData object where values are concatanation of common units values. The result
        is of shape (N, p1 + p2 + ...).

    Examples:
        >>> lifetime_data_1 = LifetimeData(values = np.array([[1], [2]]), unit_ids = np.array([3, 10]))
        >>> lifetime_data_2 = LifetimeData(values = np.array([[3], [5]]), unit_ids = np.array([10, 2]))
        >>> intersect_lifetimes(lifetime_data_1, lifetime_data_2)
        LifetimeData(values=array([[2, 3]]), unit_ids=array([10]))
    """

    inter_ids = np.array(list(set.intersection(*[set(m.unit_ids) for m in lifetimes])))
    return LifetimeData(
        np.hstack([m.values[np.isin(m.unit_ids, inter_ids)] for m in lifetimes]),
        inter_ids,
    )


def lifetimes_compatibility(
    observed_lifetimes: ObservedLifetimes, truncations: Truncations
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
        print(lifetimes)
        if len(truncations.left) != 0 and len(lifetimes) != 0:
            left_truncated_lifetimes = intersect_lifetimes(lifetimes, truncations.left)
            if len(left_truncated_lifetimes) != 0:
                if np.any(
                    np.min(
                        left_truncated_lifetimes.values[
                            :, : lifetimes.values.shape[-1]
                        ],
                        axis=1,
                        keepdims=True,
                    )
                    < left_truncated_lifetimes.values[:, lifetimes.values.shape[-1] :]
                ):
                    raise ValueError(
                        """"
                        Some lifetimes are under left truncation bounds
                        """
                    )
        if len(truncations.right) != 0 and len(lifetimes) != 0:
            left_truncated_lifetimes = intersect_lifetimes(lifetimes, truncations.right)
            if len(left_truncated_lifetimes) != 0:
                if np.any(
                    np.max(
                        left_truncated_lifetimes.values[
                            :, : lifetimes.values.shape[-1]
                        ],
                        axis=1,
                        keepdims=True,
                    )
                    > left_truncated_lifetimes.values[:, lifetimes.values.shape[-1] :]
                ):
                    raise ValueError(
                        """
                        Some lifetimes are above right truncation bounds
                        """
                    )


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
        self.entry = entry
        self.departure = departure
        self.lc_indicators = lc_indicators
        self.rc_indicators = rc_indicators

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

    def get_complete(self) -> LifetimeData:
        self.lc_indicators: BoolArray
        self.rc_indicators: BoolArray
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
    Concrete implementation of LifetimeDataFactory for 2D encoding
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
        lifetimes = LifetimeData(values, index)
        if len(lifetimes) != 0:
            if np.any(lifetimes.values[:, 0] >= lifetimes.values[:, 1]):
                raise ValueError(
                    "Interval censorships lower bounds can't be higher or equal to its upper bounds"
                )
        return lifetimes

    def get_left_truncations(self) -> LifetimeData:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        return LifetimeData(values, index)

    def get_right_truncations(self) -> LifetimeData:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        return LifetimeData(values, index)


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
