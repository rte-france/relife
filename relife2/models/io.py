from typing import Optional, Union, Sequence

import numpy as np
from numpy.typing import ArrayLike


def array_factory(
    obj: Union[ArrayLike, None], nb_units: Optional[int] = None
) -> np.ndarray:
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
    if obj is not None:
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
    time: ArrayLike,
    event: Optional[ArrayLike] = None,
    entry: Optional[ArrayLike] = None,
    departure: Optional[ArrayLike] = None,
    args: Optional[Sequence[ArrayLike] | ArrayLike] = (),
):
    time = array_factory(time)
    if time.shape[-1] not in (1, 2):
        raise ValueError(
            f"time must be of shape compatible to (nb_units, 1) or (nb_units, 2), got {time.shape}"
        )
    if event is not None:
        event = array_factory(event, nb_units=time.shape[0])
    if entry is not None:
        entry = array_factory(entry, nb_units=time.shape[0])
    if departure is not None:
        departure = array_factory(departure, nb_units=time.shape[0])

    args = [array_factory(arg, nb_units=time.shape[0]) for arg in args]

    return time, event, entry, departure, args
