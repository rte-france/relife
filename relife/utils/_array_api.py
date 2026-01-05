from typing import Any, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from relife.typing import NumpyFloat

__all__ = [
    "reshape_1d_arg",
    "flatten_if_possible",
    "get_args_nb_assets",
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
