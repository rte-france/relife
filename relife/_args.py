from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from relife import ParametricModel
    from relife.economic import CostStructure

"""
arrays cannot have more than 2 dim

if input is -> output is :
* float -> float
* 1darray, size 1 -> float
* 1darray, size m -> 2darray shape (m, 1) or (1, m)
* 2darray, size 1 -> float
* 2darray, size m*n -> 2darray shape (n, m)
"""


def args_names_generator(
    model: ParametricModel,
) -> Iterator[str]:
    from relife.lifetime_model import (
        AFT,
        AgeReplacementModel,
        LeftTruncatedModel,
        ProportionalHazard,
    )

    match model:
        case ProportionalHazard() | AFT():
            yield "covar"
            yield from args_names_generator(model.baseline)
        case AgeReplacementModel():
            yield "ar"
            yield from args_names_generator(model.baseline)
        case LeftTruncatedModel():
            yield "a0"
            yield from args_names_generator(model.baseline)
        case _:  # in other case, stop generator and yield nothing
            return


def reshape_arg(
    name: str, value: float | NDArray[np.float64]
) -> float | NDArray[np.float64]:
    value = np.asarray(value)
    ndim = value.ndim
    if ndim > 2:
        raise ValueError(
            f"Number of dimension can't be higher than 2. Got {ndim} for {name}"
        )
    match name:
        case "covar":
            if value.ndim <= 1:
                return value.reshape(1, -1)
            return value
        case "a0" | "ar" | "ar1" | "cf" | "cp" | "cr":
            if ndim <= 2:
                return value.reshape(-1, 1)
            raise ValueError(f"{name} arg can't have more than 1 dim")


def get_nb_assets(*args: float | NDArray[np.float64]) -> int:
    def get_nb_asset(x: float | NDArray[np.float64]):
        if isinstance(x, np.ndarray):
            return x.shape[0]
        else:
            return 1

    return max(map(lambda x: get_nb_asset(x), args), default=1)


def broadcast_args(
    obj: Union[ParametricModel, CostStructure],
    *args: float | NDArray[np.float64],
    **kwargs: float | NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:

    args = args + tuple(kwargs.values())
    args_names = tuple(args_names_generator(obj)) + tuple(kwargs.keys())
    if len(args_names) != len(args):
        raise TypeError(
            f"{obj.__class__.__name__} requires {args_names} positional argument but got {len(args)} argument.s only"
        )
    new_args = tuple((reshape_arg(k, v) for k,v in zip(args_names, args)))
    try :
        np.broadcast_shapes(*tuple((arg.shape for arg in new_args)))
    except ValueError as err:
        raise ValueError("Unbroadcastable args") from err
    return {k : v for k, v in zip(args_names, new_args)}
