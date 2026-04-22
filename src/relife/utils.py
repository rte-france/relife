from typing import Literal, TypeAlias, overload

import numpy as np
from optype.numpy import (
    Array,
    Array0D,
    Array1D,
    Array2D,
    ArrayND,
    is_array_0d,
)

__all__ = [
    "to_column_2d_if_1d",
    "to_numpy_float64",
    "flatten_if_at_least_2d",
    "get_nb_assets",
]


@overload
def to_numpy_float64(v: float | np.floating | np.uint) -> np.float64: ...
@overload
def to_numpy_float64(
    v: ArrayND[np.floating | np.uint],
) -> ArrayND[np.float64]: ...
def to_numpy_float64(
    v: float | np.uint | np.floating | ArrayND[np.floating | np.uint],
) -> np.float64 | ArrayND[np.float64]:
    """
    Convert the input to np.float64 if it is a scalar or an array of np.float64
    otherwise.
    """
    if isinstance(v, (int, float)):
        return np.float64(v)
    return np.asarray(v, dtype=np.float64)


ST: TypeAlias = int | float | bool
NumpyST: TypeAlias = np.floating | np.uint | np.bool


# type checkers warn about overload 2 because bool is subclass of int in Python
# that's the best we can do.
@overload
def to_column_2d_if_1d(
    arg: int | float,
) -> np.float64: ...
@overload
def to_column_2d_if_1d(
    arg: bool,
) -> np.bool: ...
@overload
def to_column_2d_if_1d(
    arg: np.floating | np.uint,
) -> np.float64: ...
@overload
def to_column_2d_if_1d(
    arg: np.bool,
) -> np.bool: ...
@overload
def to_column_2d_if_1d(
    arg: Array0D[np.floating | np.uint],
) -> Array0D[np.float64]: ...
@overload
def to_column_2d_if_1d(
    arg: Array0D[np.bool],
) -> Array0D[np.bool]: ...
@overload
def to_column_2d_if_1d(
    arg: Array1D[np.floating | np.uint],
) -> Array[tuple[int, Literal[1]], np.float64]: ...
@overload
def to_column_2d_if_1d(
    arg: Array1D[np.bool],
) -> Array[tuple[int, Literal[1]], np.bool]: ...
@overload
def to_column_2d_if_1d(arg: Array2D[np.floating | np.uint]) -> Array2D[np.float64]: ...
@overload
def to_column_2d_if_1d(arg: Array2D[np.bool]) -> Array2D[np.bool]: ...
@overload
def to_column_2d_if_1d(
    arg: Array[tuple[int, int, *tuple[int, ...]], np.floating | np.uint],
) -> Array[tuple[int, int, *tuple[int, ...]], np.float64]: ...
@overload
def to_column_2d_if_1d(
    arg: Array[tuple[int, int, *tuple[int, ...]], np.bool],
) -> Array[tuple[int, int, *tuple[int, ...]], np.bool]: ...
def to_column_2d_if_1d(
    arg: ST
    | NumpyST
    | Array0D[NumpyST]
    | Array1D[NumpyST]
    | Array2D[NumpyST]
    | Array[tuple[int, int, *tuple[int, ...]], NumpyST],
) -> (
    np.float64
    | np.bool
    | Array0D[np.float64]
    | Array0D[np.bool]
    | Array[tuple[int, Literal[1]], np.float64]
    | Array2D[np.float64]
    | Array[tuple[int, int, *tuple[int, ...]], np.float64]
):
    """
    Reshapes to column array.

    Parameters
    ----------
    arg : any scalar or array

    Returns
    -------
    np.ndarray
    """
    if isinstance(arg, np.ndarray):
        dtype = np.float64 if arg.dtype != np.bool else np.bool
    else:
        dtype = np.float64 if not isinstance(arg, bool) else np.bool
    arr = np.asarray(arg, dtype=dtype)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


@overload
def flatten_if_at_least_2d(value: ST | NumpyST) -> np.float64: ...
@overload
def flatten_if_at_least_2d(value: Array0D[NumpyST]) -> Array0D[np.float64]: ...
@overload
def flatten_if_at_least_2d(value: Array1D[NumpyST]) -> Array1D[np.float64]: ...
@overload
def flatten_if_at_least_2d(
    value: Array[tuple[int, int, *tuple[int, ...]], NumpyST],
) -> Array1D[np.float64]: ...
def flatten_if_at_least_2d(
    value: ST
    | NumpyST
    | Array0D[NumpyST]
    | Array1D[NumpyST]
    | Array[tuple[int, int, *tuple[int, ...]], NumpyST],
) -> np.float64 | Array0D[np.float64] | Array1D[np.float64]:
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
    if isinstance(value, np.ndarray):
        if value.ndim >= 1:
            return value.astype(np.float64).flatten()
        # typeguards
        assert is_array_0d(value)
        return value.astype(np.float64)
    return np.float64(value)


def get_nb_assets(*args: ST | NumpyST | ArrayND[NumpyST]) -> int:
    """
    Gets the number of assets encoded in args.
    """
    if not bool(args):
        return 1
    reshaped_args = tuple(np.atleast_2d(arg) for arg in args)
    try:
        broadcast_shape = np.broadcast_shapes(*(ary.shape for ary in reshaped_args))
    except ValueError as err:
        raise ValueError("args have incompatible shapes") from err
    if len(broadcast_shape) == 0:
        return 1
    return broadcast_shape[0]
