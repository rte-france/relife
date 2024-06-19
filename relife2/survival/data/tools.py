import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from relife2.survival.data.lifetimes import LifetimeData, ObservedLifetimes, Truncations

FloatArray = NDArray[np.float64]


def intersect_lifetimes(*lifetimes: LifetimeData) -> LifetimeData:
    """
    Args:
        *lifetimes: LifetimeData object.s containing values of shape (n1, p1), (n2, p2), etc.

    Returns:
        LifetimeData: One LifetimeData object where values are concatanation of common units values. The result
        is of shape (N, p1 + p2 + ...).

    Examples:
        >>> lifetime_data_1 = LifetimeData(values = np.array([[1], [2]]), index = np.array([3, 10]))
        >>> lifetime_data_2 = LifetimeData(values = np.array([[3], [5]]), index = np.array([10, 2]))
        >>> intersect_lifetimes(lifetime_data_1, lifetime_data_2)
        LifetimeData(values=array([[2, 3]]), unit_ids=array([10]))
    """

    inter_ids = np.array(list(set.intersection(*[set(m.index) for m in lifetimes])))
    return LifetimeData(
        np.hstack([m.values[np.isin(m.index, inter_ids)] for m in lifetimes]),
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
