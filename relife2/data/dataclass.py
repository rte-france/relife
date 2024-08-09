"""
This module defines dataclass used to encapsulate data used for parameters estimation of models

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from relife2.typing import FloatArray, IntArray


@dataclass
class Sample:
    """
    Object that encapsulates lifetime data values and corresponding units index
    """

    values: FloatArray
    ids: IntArray
    extravars: dict[str, Any] = field(
        default_factory=dict
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
        )


@dataclass
class Truncations:
    """BLABLABLA"""

    left: Sample
    right: Sample


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
