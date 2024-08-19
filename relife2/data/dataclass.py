"""
This module defines dataclass used to encapsulate data used for parameters estimation of models

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sample:
    """
    Object that encapsulates lifetime data values and corresponding units index
    """

    values: np.ndarray
    ids: np.ndarray  # arrays of int
    args: list[np.ndarray] = field(
        default_factory=list
    )  # any other data attached to values (e.g. covar values)

    def __post_init__(self):
        if self.values.ndim != 2:
            raise ValueError("Invalid LifetimeData values number of dimensions")
        if self.ids.ndim != 1:
            raise ValueError("Invalid LifetimeData unit_ids number of dimensions")
        if len(self.values) != len(self.ids):
            raise ValueError("Incompatible lifetime values and unit_ids")
        if np.all(self.values == 0, axis=1).any():
            raise ValueError("Lifetimes values must be greater than 0")

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class LifetimeSample:
    """BLABLABLA"""

    complete: Sample
    left_censored: Sample
    right_censored: Sample
    interval_censored: Sample

    def __post_init__(self):
        self.rc = Sample(
            np.concatenate(
                (
                    self.complete.values,
                    self.right_censored.values,
                ),
                axis=0,
            ),
            np.concatenate((self.complete.ids, self.right_censored.ids)),
            [
                np.concatenate(
                    (v, self.right_censored.args[i]),
                    axis=0,
                )
                for i, v in enumerate(self.complete.args)
            ],
        )
        self.rlc = Sample(
            np.concatenate(
                [
                    self.complete.values,
                    self.left_censored.values,
                    self.right_censored.values,
                ]
            ),
            np.concatenate(
                (
                    self.complete.ids,
                    self.left_censored.ids,
                    self.right_censored.ids,
                )
            ),
            [
                np.concatenate(
                    (
                        v,
                        self.left_censored.args[i],
                        self.right_censored.args[i],
                    ),
                    axis=0,
                )
                for i, v in enumerate(self.complete.args)
            ],
        )


def intersect_lifetimes(*lifetimes: Sample) -> list[Sample]:
    """
    Args:
        *lifetimes: LifetimeData object.s containing values of shape (n1, p1), (n2, p2), etc.

    Returns:

    Examples:
        >>> lifetimes_1 = Sample(values = np.array([[1], [2]]), ids = np.array([3, 10]))
        >>> lifetimes_2 = Sample(values = np.array([[3], [5]]), ids = np.array([10, 2]))
        >>> intersect_lifetimes(lifetimes_1, lifetimes_2)
        [Lifetimes(values=array([[2]]), index=array([10])), Lifetimes(values=array([[3]]), index=array([10]))]
    """

    inter_ids = np.array(
        list(set.intersection(*[set(_lifetimes.ids) for _lifetimes in lifetimes]))
    )
    return [
        Sample(
            _lifetimes.values[np.isin(_lifetimes.ids, inter_ids)],
            inter_ids,
            [v[np.isin(_lifetimes.ids, inter_ids)] for v in _lifetimes.args],
        )
        for _lifetimes in lifetimes
    ]


@dataclass
class Truncations:
    """BLABLABLA"""

    left: Sample
    right: Sample


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
