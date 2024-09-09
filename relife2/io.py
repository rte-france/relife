from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike


def get_nb_units(obj: Union[np.ndarray, None]) -> int:
    try:
        obj = np.asarray(obj, dtype=np.float64)
    except Exception as error:
        raise ValueError(
            "Invalid type of arg : can't be cast to np.ndarray with dtype float64"
        ) from error
    if obj.ndim <= 1:
        return 1
    elif obj.ndim == 2:
        return obj.shape[0]
    else:
        raise ValueError(
            f"Input arg can't have more than 2 dimensions : got shape {obj.shape}"
        )


def array_factory(obj: np.ndarray, nb_units: Optional[int] = None) -> np.ndarray:
    """
    Converts object input to 2d array of shape (n, p)
    n is the number of units
    p is the number of points
    Args:
        nb_units ():
        obj: object input

    Returns:
        np.ndarray: 2d array
    """
    try:
        obj = np.asarray(obj, dtype=np.float64)
    except Exception as error:
        raise ValueError(
            "Invalid type of arg : can't be cast to np.ndarray with dtype float64"
        ) from error
    if obj.ndim <= 1:
        if nb_units is not None:
            try:
                obj = obj.reshape(nb_units, -1)
            except ValueError:
                raise ValueError(
                    f"Invalid arg shape : can't reshape arg with {nb_units} as first dim"
                )
        else:
            obj = obj.reshape(-1, 1)
    elif obj.ndim == 2:
        if nb_units is not None:
            if obj.shape[0] != nb_units:
                raise ValueError(
                    f"Invalid arg shape : expected first dim as {nb_units} but got {obj.shape[0]}"
                )
    else:
        raise ValueError(
            f"Input arg can't have more than 2 dimensions : got shape {obj.shape}"
        )
    return obj


def are_all_args_given(function, *args: ArrayLike):
    if len(args) != function.nb_args:
        raise ValueError(
            f"Model function expects {function.nb_args} args but got {len(args)} : required args are {function.args_names}"
        )


def preprocess_lifetime_data(
    time: np.ndarray,
    event: Optional[np.ndarray] = None,
    entry: Optional[np.ndarray] = None,
    departure: Optional[np.ndarray] = None,
    args: tuple[np.ndarray, ...] | tuple[()] = (),
):
    time = array_factory(time)
    if event is not None:
        event = array_factory(event, nb_units=time.shape[0])
    if entry is not None:
        entry = array_factory(entry, nb_units=time.shape[0])
    if departure is not None:
        departure = array_factory(departure, nb_units=time.shape[0])
    args = tuple([array_factory(arg, nb_units=time.shape[0]) for arg in args])
    return time, event, entry, departure, args
