from typing import Callable

import numpy as np

from ...arrays.inference import has_same_length, is_1d, is_2d, is_array

"""
For y = f(x)
    one check if x.shape == y.shape and x.ndim == y.ndim == 1 (hf, sf, chf, etc.)
    one check if x.ndim == 1 and y == float (mean, var, etc.)
    one check if x.ndim == 1 and y.ndim == 2 and y.shape == (x.shape[0], nb_param) (jac_hf, jac, etc.)
"""


def check_func_to_1d_array(f: Callable, *args):
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
        out_shape = out.shape
        raise ValueError(
            f"""1d array output from {f.__name__} expected but got  {out_ndim} dim with {out_shape} shape"""
        ) from error
    try:
        has_same_length(out, test_input)
    except Exception as error:
        raise ValueError(
            f"input and output of {f.__name__} must have the same length"
        ) from error


def func_1d_array_2d_array(f: Callable, *args):
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
        is_2d(out)
    except Exception as error:
        out_ndim = out.ndim
        out_shape = out.shape
        raise ValueError(
            f"""2d array output from {f.__name__} expected but got  {out_ndim} dim with {out_shape} shape"""
        ) from error
    try:
        has_same_length(out, test_input)
    except Exception as error:
        raise ValueError(
            f"input and output of {f.__name__} must have the same length"
        ) from error


def func_1d_array_to_(f: Callable, *args):
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
        is_2d(out)
    except Exception as error:
        out_ndim = out.ndim
        out_shape = out.shape
        raise ValueError(
            f"""2d array output from {f.__name__} expected but got  {out_ndim} dim with {out_shape} shape"""
        ) from error
    try:
        has_same_length(out, test_input)
    except Exception as error:
        raise ValueError(
            f"input and output of {f.__name__} must have the same length"
        ) from error
