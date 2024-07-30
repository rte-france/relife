"""
This module defines tools functions used to treat statistical data

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import numpy as np
from numpy.typing import ArrayLike

from relife2.data.dataclass import Lifetimes, ObservedLifetimes, Truncations
from relife2.utils.types import FloatArray


def intersect_lifetimes(*lifetimes: Lifetimes) -> tuple[Lifetimes]:
    """
    Args:
        *lifetimes: LifetimeData object.s containing values of shape (n1, p1), (n2, p2), etc.

    Returns:

    Examples:
        >>> lifetimes_1 = Lifetimes(values = np.array([[1], [2]]), index = np.array([3, 10]))
        >>> lifetimes_2 = Lifetimes(values = np.array([[3], [5]]), index = np.array([10, 2]))
        >>> intersect_lifetimes(lifetimes_1, lifetimes_2)
        (Lifetimes(values=array([[2]]), index=array([10])), Lifetimes(values=array([[3]]), index=array([10])))
    """

    inter_ids = np.array(
        list(set.intersection(*[set(_lifetimes.index) for _lifetimes in lifetimes]))
    )
    return tuple(
        [
            Lifetimes(
                _lifetimes.values[np.isin(_lifetimes.index, inter_ids)], inter_ids
            )
            for _lifetimes in lifetimes
        ]
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
