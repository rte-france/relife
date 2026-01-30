from typing import Any, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from relife.typing import NumpyFloat

__all__ = [
    "reshape_1d_arg",
    "flatten_if_possible",
    "get_args_nb_assets",
    "is_2d_np_array",
    "nearest_1dinterp",
    "get_ordered_event_time"
]


E = TypeVar("E", bound=np.generic, covariant=True)


@overload
def reshape_1d_arg(arg: int) -> np.float64: ...
@overload
def reshape_1d_arg(arg: float) -> np.float64: ...
@overload
def reshape_1d_arg(arg: bool | np.bool_) -> np.bool: ...
@overload
def reshape_1d_arg(arg: NDArray[E]) -> NDArray[E]: ...
def reshape_1d_arg(arg: int | float | bool | np.bool | NDArray[E]) -> np.float64 | np.bool | NDArray[E]:
    """
    Reshapes ReLife arguments that are expected to be 0d or 1d.

    Parameters
    ----------
    arg : float, 0d or 1d array

    Returns
    -------
    np.float64 or (m, 1) shaped array
        Reshaped array used to ensure broadcasting compatibility in computations.
    """
    if isinstance(arg, (int, float)):  # np.float64 is float
        return np.float64(arg)
    if isinstance(arg, (bool, np.bool)):
        return np.bool(arg)
    if arg.ndim == 1:
        arg = np.atleast_1d(arg).reshape(-1, 1)
    if arg.ndim > 2:
        raise ValueError("arg can't be more than 2d")
    return arg


def is_2d_np_array(arr) -> bool:
    """
    Boolean test for 2d numpy array
    """
    if not isinstance(arr, np.ndarray):
        return False
    return arr.ndim == 2


def nearest_1dinterp(x: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """Returns x nearest interpolation based on xp and yp data points
    xp has to be monotonically increasing

    Args:
        x (np.ndarray): 1d x coordinates to interpolate
        xp (np.ndarray): 1d known x coordinates
        yp (np.ndarray): 1d known y coordinates

    Returns:
        np.ndarray: interpolation values of x
    """
    spacing = np.diff(xp) / 2
    xp = xp + np.hstack([spacing, spacing[-1]])
    yp = np.concatenate([yp, yp[-1, None]])
    return yp[np.searchsorted(xp, x)]


def get_ordered_event_time(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From time and event samples returns uncensored sorted untied times,
    associated original index and counts

     Args:
        time (np.ndarray): 1d or reshape_1d_arg time sample
        event (np.ndarray): 1d or reshape_1d_arg event sample

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: uncensored sorted untied times,
        associated original index and counts
    """
    (
        ordered_event_time,  # uncensored sorted untied times
        ordered_event_index,
        event_count,
    ) = np.unique(
        time[event == 1],
        return_index=True,
        return_counts=True,
    )
    return ordered_event_time, ordered_event_index, event_count


def flatten_if_possible(value: NumpyFloat) -> NumpyFloat:
    """
    Flatten array-like object when possible.

    Parameters
    ----------
    value : np.ndarray

    Returns
    -------
    np.ndarray
        Flattened array.
    """
    if value.ndim != 0:
        return value.flatten()
    return value


def get_args_nb_assets(*args: NDArray[Any]) -> int:
    """
    Gets the number of assets encoded in args.
    """
    if not bool(args):
        return 1
    reshaped_args = tuple((np.atleast_2d(arg) for arg in args))
    try:
        broadcast_shape = np.broadcast_shapes(*(ary.shape for ary in reshaped_args))
    except ValueError:
        raise ValueError("args have incompatible shapes")
    if len(broadcast_shape) == 0:
        return 1
    return broadcast_shape[0]
