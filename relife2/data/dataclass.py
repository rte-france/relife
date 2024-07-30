"""
This module defines dataclass used to encapsulate data used for parameters estimation of models

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from dataclasses import dataclass

import numpy as np

from relife2.utils.types import FloatArray, IntArray


@dataclass(frozen=True)
class Lifetimes:
    """
    Object that encapsulates lifetime data values and corresponding units index
    """

    values: FloatArray
    index: IntArray

    def __post_init__(self):
        if self.values.ndim != 2:
            raise ValueError("Invalid LifetimeData values number of dimensions")
        if self.index.ndim != 1:
            raise ValueError("Invalid LifetimeData unit_ids number of dimensions")
        if len(self.values) != len(self.index):
            raise ValueError("Incompatible lifetime values and unit_ids")
        if np.all(self.values == 0, axis=1).any():
            raise ValueError("Lifetimes values must be greater than 0")

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class ObservedLifetimes:
    """BLABLABLA"""

    complete: Lifetimes
    left_censored: Lifetimes
    right_censored: Lifetimes
    interval_censored: Lifetimes

    def __post_init__(self):
        self.rc = Lifetimes(
            np.concatenate(
                (
                    self.complete.values,
                    self.right_censored.values,
                ),
                axis=0,
            ),
            np.concatenate((self.complete.index, self.right_censored.index)),
        )
        self.rlc = Lifetimes(
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


@dataclass(frozen=True)
class Truncations:
    """BLABLABLA"""

    left: Lifetimes
    right: Lifetimes


@dataclass
class Deteriorations:
    """BLABLABLA"""

    values: FloatArray  # R0 in first column (always)
    times: FloatArray  # 0 in first column (always)
    ids: IntArray

    def __post_init__(self):
        # self.values = np.ma.array(self.values, mask=np.isnan(self.values))
        # self.times = np.ma.array(self.times, mask=np.isnan(self.times))
        self.increments = -np.diff(self.values, axis=1)
        self.event = self.increments == 0
