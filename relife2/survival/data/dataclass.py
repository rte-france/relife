from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ExtraData:
    values: FloatArray
    index: IntArray

    values: FloatArray
    index: IntArray

    def __post_init__(self):
        if self.values.ndim != 2:
            raise ValueError("Invalid LifetimeData values number of dimensions")
        if self.index.ndim != 1:
            raise ValueError("Invalid LifetimeData unit_ids number of dimensions")
        if len(self.values) != len(self.index):
            raise ValueError("Incompatible lifetime values and unit_ids")

    def __len__(self) -> int:
        return len(self.values)


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
