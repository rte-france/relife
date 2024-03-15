from typing import Any, Tuple

import numpy as np


def is_array(*x: Any):
    if type(x) == tuple:
        types = {type(_x) for _x in x}
        if types != {np.ndarray}:
            raise TypeError(f"Expected only np.ndarray, got {types}")
    else:
        if not type(x) == np.ndarray:
            raise TypeError(f"Expected np.array, got '{type(x).__name__}'")


def is_instance_of(*x: Any, parent_obj: Any):
    if type(x) == tuple:
        instances = {isinstance(_x, parent_obj) for _x in x}
        if instances != {True}:
            raise ValueError(f"Expected only {parent_obj} instance")
    else:
        if not isinstance(x, parent_obj):
            raise ValueError(f"Expected {parent_obj} instance")


def is_0d(x: np.ndarray):
    if not x.ndim == 0:
        ndim = x.ndim
        raise ValueError(f"Expected 0 dim array but got {ndim}")


def is_1d(x: np.ndarray):
    if not x.ndim == 1:
        ndim = x.ndim
        raise ValueError(f"Expected 1 dim array but got {ndim}")


def is_2d(x: np.ndarray):
    ndim = x.ndim
    shape = x.shape
    if not ndim == 2:
        raise ValueError(f"Expected 2 dim array but got {ndim}")
    elif not shape[-1] == 2:
        raise ValueError(
            f"Expected array of shape (n, 2) but got shape {shape}"
        )


def is_1d_or_2d(x: np.ndarray):
    ndim = x.ndim
    shape = x.shape
    if not ndim == 1 and not ndim == 2:
        raise ValueError(f"Expected 2 or 1 dim array but got {ndim}")
    elif ndim == 2:
        if not shape[-1] == 2:
            raise ValueError(
                f"Expected array of shape (n, 2) but got shape {shape}"
            )


def has_same_length(*x: Tuple[np.ndarray]):
    lengths = {len(_x) for _x in x}
    if len(lengths) != 1:
        raise ValueError(
            f"Expected array of same length but got {lengths} lengths"
        )


def is_array_type(x: np.ndarray, np_type: np.dtype):
    x_type = x.dtype
    if not x_type == np_type:
        raise TypeError(f"Expected {np_type} array, got '{x_type}' array")
