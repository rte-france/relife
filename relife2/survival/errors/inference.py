from typing import Any, Callable

import numpy as np


def is_float(x: Any):
    if type(x) != float:
        TypeError(f"Expected float, got '{type(x).__name__}'")


def is_int(x: Any):
    if type(x) != int:
        TypeError(f"Expected int, got '{type(x).__name__}'")


def is_array(x: Any):
    if not type(x) != np.ndarray:
        raise TypeError(f"Expected np.array, got '{type(x).__name__}'")


def check_shape(shape: tuple, shape_value: int, axis: int = 0):
    if len(shape) - 1 < axis:
        raise ValueError(f"Array has no axis {axis}")
    if shape[axis] != shape_value:
        shape_value_on_axis = shape[axis]
        raise ValueError(
            f"Shape is {shape_value_on_axis} on axis {axis} but expected"
            f" {shape_value}"
        )


def same_shape(*x: np.ndarray):
    arrays_shapes = {_x.shape for _x in x}
    if not len(arrays_shapes) == 1:
        raise ValueError("Arrays do not have the same shapes")


def is_0d(x: np.ndarray):
    if not x.ndim == 0:
        ndim = x.ndim
        raise ValueError(f"Expected 0 dim array but got {ndim}")


def is_1d(x: np.ndarray):
    if not x.ndim == 1:
        ndim = x.ndim
        raise ValueError(f"Expected 1 dim array but got {ndim}")


def is_2d(x: np.ndarray):
    if not x.ndim == 2:
        ndim = x.ndim
        raise ValueError(f"Expected 2 dim array but got {ndim}")


def is_1d_or_2d(x: np.ndarray):
    ndim = x.ndim
    if not ndim == 1 and not ndim == 2:
        raise ValueError(f"Expected 2 or 1 dim array but got {ndim}")


def is_array_type(x: np.ndarray, np_type: np.dtype):
    x_type = x.dtype
    if not x_type == np_type:
        raise TypeError(f"Expected {np_type} array, got '{x_type}' array")


float_func = ["mean", "var"]


def check_float_func(f: Callable, *args):
    """
    For y = f(x)
    one check if x.ndim == 1 and y == float (mean, var, etc.)
    """
    test_input = np.array([1, 2])
    try:
        out = f(test_input, *args)
    except Exception as error:
        raise RuntimeError(
            f"{f.__name__} function does not work as expected"
        ) from error
    try:
        is_float(out)
    except Exception as error:
        out_type = type(out)
        raise ValueError(
            f"Expected float output from {f.__name__} but got {out_type}"
        ) from error


array_1d_func = ["", ""]


def check_array_1d_func(f: Callable, *args):
    """
    For y = f(x)
    one check if x.shape == y.shape and x.ndim == y.ndim == 1 (hf, sf, chf, etc.)
    """
    test_input = np.array([1, 2])
    try:
        out = f(test_input, *args)
    except Exception as error:
        raise RuntimeError(
            f"{f.__name__} function does not work as expected"
        ) from error
    try:
        is_array(out)
    except Exception as error:
        out_type = type(out)
        raise ValueError(
            f"Expected array output from {f.__name__} but got {out_type}"
        ) from error
    try:
        is_1d(out)
    except Exception as error:
        out_ndim = out.ndim
        raise ValueError(
            f"""1d array output from {f.__name__} expected but got {out_ndim} dim array"""
        ) from error
    try:
        check_shape(out.shape, len(test_input), 0)
    except Exception as error:
        raise ValueError(
            f"input and output of {f.__name__} must have the same length"
        ) from error


jac_func = ["", ""]


def check_jac_func(f: Callable, nb_params: int, *args):
    """
    For y = f(x)
    one check if x.ndim == 1 and y.ndim == 2 and y.shape == (x.shape[0], nb_param) (jac_hf, jac, etc.)
    """
    test_input = np.array([1, 2])
    try:
        out = f(test_input, *args)
    except Exception as error:
        raise RuntimeError(
            f"{f.__name__} function does not work as expected"
        ) from error
    try:
        is_array(out)
    except Exception as error:
        out_type = type(out)
        raise ValueError(
            f"Expected array output from {f.__name__} but got {out_type}"
        ) from error
    try:
        check_shape(out.shape, len(test_input), 0)
        check_shape(out.shape, nb_params, 1)
    except Exception as error:
        out_shape = out.shape
        raise ValueError(
            f"""Shape output {out_shape} from {f.__name__} invalid"""
        ) from error
