from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    overload,
)

import numpy as np
from numpy.typing import NDArray


@overload
def check_and_broadcast_bounds(
    a: float | NDArray[np.float64],
    b: None = None,
) -> NDArray[np.float64]: ...


@overload
def check_and_broadcast_bounds(
    a: float | NDArray[np.float64],
    b: float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


def check_and_broadcast_bounds(
    a: float | NDArray[np.float64],
    b: Optional[float | NDArray[np.float64]] = None,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:

    def control_shape(bound: float | NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(bound, dtype=np.float64)
        if np.any(arr < 0):
            raise ValueError("Bound values of the integral can't be lower than 0")
        if arr.ndim > 2:
            raise ValueError("Bound the integral can't have more than 2 dimensions")
        return arr

    a = control_shape(a)
    if b is not None:
        b = control_shape(b)
        try:
            a, b = np.broadcast_arrays(a, b)
            return a.copy(), b.copy()
        except ValueError as err:
            raise ValueError(f"Incompatible a, b shapes. Got a.shape, b.shape : {a.shape}, {b.shape}") from err
    return a


def legendre_quadrature(
    func: Callable[[float | NDArray[np.float64]], np.float64 | NDArray[np.float64]],
    a: float | NDArray[np.float64],
    b: float | NDArray[np.float64],
    deg: int = 10,
) -> np.float64 | NDArray[np.float64]:
    r"""Numerical integration of `func` over the interval `[a,b]`

    `func` must accept (deg,), (deg, n) or (deg, m, n) array shapes
    a can be zero
    b must not be inf

    a, b shapes can be either 0d (float like), 1d or 2d
    """
    arr_a, arr_b = check_and_broadcast_bounds(a, b)  # () or (n,) or (m, n)
    quad = np.polynomial.legendre.leggauss(deg)  # (deg,)
    x = quad[0].reshape((-1,) + (1,) * arr_a.ndim)  # (deg,), (deg, 1) or (deg, 1, 1)
    w = quad[1].reshape((-1,) + (1,) * arr_a.ndim)  # (deg,), (deg, 1) or (deg, 1, 1)

    if np.any(arr_b == np.inf):
        raise ValueError("Bound values of Legendre quadrature must be finite")
    if np.any(arr_a > arr_b):
        raise ValueError("Bound values a must be lower than values of b")

    p = (arr_b - arr_a) / 2  # () or (n,) or (m, n)
    m = (arr_a + arr_b) / 2  # () or (n,) or (m, n)
    u = p * x + m  # (deg,) or (deg, n) or (deg, m, n)
    v = p * w  # (deg,) or (deg, n) or (deg, m, n)
    fvalues = func(u)  # (d_1, ..., d_i, deg) or (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
    if fvalues.shape[-len(u.shape) :] != u.shape:
        raise ValueError(
            f"""
            func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
            Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
            """
        )

    return np.sum(v * fvalues, axis=-v.ndim)  # (d_1, ..., d_i) or (d_1, ..., d_i, n) or (d_1, ..., d_i, m, n)


def laguerre_quadrature(
    func: Callable[[float | NDArray[np.float64]], np.float64 | NDArray[np.float64]],
    a: float | NDArray[np.float64] = 0.0,
    deg: int = 10,
) -> np.float64 | NDArray[np.float64]:
    r"""Numerical integration of `func * exp(-x)` over the interval `[a, inf]`

    `func` must accept (deg,), (deg, n) or (deg, m, n) array shapes
    It must handle at least 3 dimensions.
    a can be zero with ndim <= 2.
    """
    arr_a = check_and_broadcast_bounds(a)  # () or (n,) or (m, n)
    quad = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    x = quad[0].reshape((-1,) + (1,) * arr_a.ndim)  # (deg,), (deg, 1) or (deg, 1, 1)
    w = quad[1].reshape((-1,) + (1,) * arr_a.ndim)  # (deg,), (deg, 1) or (deg, 1, 1)

    shifted_x = x + arr_a  # (deg,) or (deg, n) or (deg, m, n)
    fvalues = func(shifted_x)  # (d_1, ..., d_i, deg) or (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
    if fvalues.shape[-len(shifted_x.shape) :] != shifted_x.shape:
        # func est une fonction réel univariée et pas multivariée
        raise ValueError(
            f"""
            func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
            Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
            """
        )

    exp_a = np.where(np.exp(-arr_a) == 0, 1.0, np.exp(-arr_a))  # () or (n,) or (m, n)
    return np.sum(
        w * fvalues * exp_a, axis=-shifted_x.ndim
    )  # (d_1, ..., d_i) or (d_1, ..., d_i, n) or (d_1, ..., d_i, m, n)


def unweighted_laguerre_quadrature(
    func: Callable[[float | NDArray[np.float64]], np.float64 | NDArray[np.float64]],
    a: float | NDArray[np.float64] = 0.0,
    deg: int = 10,
) -> np.float64 | NDArray[np.float64]:
    r"""Numerical integration of `func` over the interval `[a, inf]`

    `func` must accept (deg,), (deg, n) or (deg, m, n) array shapes
    It must handle at least 3 dimensions.
    a can be zero with ndim <= 2.
    """

    quad = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    arr_a = check_and_broadcast_bounds(a)  # () or (n,) or (m, n)
    x = quad[0].reshape((-1,) + (1,) * arr_a.ndim)  # (deg,), (deg, 1) or (deg, 1, 1)
    w = quad[1].reshape((-1,) + (1,) * arr_a.ndim)  # (deg,), (deg, 1) or (deg, 1, 1)

    shifted_x = x + arr_a  # (deg,) or (deg, n) or (deg, m, n)
    fvalues = func(shifted_x)  # (d_1, ..., d_i, deg) or (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
    if fvalues.shape[-len(shifted_x.shape) :] != shifted_x.shape:
        raise ValueError(
            f"""
            func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
            Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
            """
        )
    return np.sum(
        w * fvalues * np.exp(x), axis=-shifted_x.ndim
    )  # (d_1, ..., d_i) or (d_1, ..., d_i, n) or (d_1, ..., d_i, m, n)
