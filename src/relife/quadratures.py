from collections.abc import Callable
from typing import TypeAlias, overload

import numpy as np
from optype.numpy import Array, ArrayND, AtMost2D

__all__ = [
    "legendre_quadrature",
    "laguerre_quadrature",
    "unweighted_laguerre_quadrature",
    "broadcast_bounds",
]

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


def _control_bound(
    bound: NumpyST | Array[AtMost2D, NumpyST],
) -> NumpyST | Array[AtMost2D, NumpyST]:
    if np.any(bound < 0):
        raise ValueError("Bound values of the integral can't be lower than 0")
    if bound.ndim > 2:
        raise ValueError("Bound the integral can't have more than 2 dimensions")
    return bound


@overload
def broadcast_bounds(
    a: NumpyST | Array[AtMost2D, NumpyST],
) -> Array[AtMost2D, NumpyST]: ...
@overload
def broadcast_bounds(
    a: NumpyST | Array[AtMost2D, NumpyST],
    b: NumpyST | Array[AtMost2D, NumpyST],
) -> tuple[Array[AtMost2D, NumpyST], Array[AtMost2D, NumpyST]]: ...
def broadcast_bounds(
    a: NumpyST | Array[AtMost2D, NumpyST],
    b: NumpyST | Array[AtMost2D, NumpyST] | None = None,
) -> (
    Array[AtMost2D, NumpyST] | tuple[Array[AtMost2D, NumpyST], Array[AtMost2D, NumpyST]]
):
    a = _control_bound(a)
    if b is not None:
        b = _control_bound(b)
        try:
            a, b = np.broadcast_arrays(a, b)
            return a.copy(), b.copy()
        except ValueError as err:
            raise ValueError("Impossible to broadcast a and b bounds.") from err
    return np.asarray(a)


def legendre_quadrature(
    func: Callable[
        [ST | NumpyST | ArrayND[NumpyST]],
        np.float64 | ArrayND[np.float64],
    ],
    a: ST | NumpyST | Array[AtMost2D, NumpyST],
    b: ST | NumpyST | Array[AtMost2D, NumpyST],
    deg: int = 10,
) -> np.float64 | Array[AtMost2D, np.float64]:
    r"""Numerical integration of :math:`f(x)` over the interval :math:`[a,b]`

    `func` must accept (deg,), (deg, n) or (deg, m, n) array shapes
    a can be zero
    b must not be inf

    a, b shapes can be either 0d (float like), 1d or 2d
    """
    arr_a, arr_b = broadcast_bounds(
        np.asarray(a), np.asarray(b)
    )  # () or (n,) or (m, n)
    quad = np.polynomial.legendre.leggauss(deg)  # (deg,)
    x, w = quad
    x = np.expand_dims(
        x, axis=tuple(range(1, arr_a.ndim + 1))
    )  # (deg,), (deg, 1) or (deg, 1, 1)
    w = np.expand_dims(
        w, axis=tuple(range(1, arr_a.ndim + 1))
    )  # (deg,), (deg, 1) or (deg, 1, 1)

    if np.any(arr_b == np.inf):
        raise ValueError("Bound values of Legendre quadrature must be finite")
    if np.any(arr_a > arr_b):
        raise ValueError("Bound values a must be lower than values of b")

    p = (arr_b - arr_a) / 2  # () or (n,) or (m, n)
    m = (arr_a + arr_b) / 2  # () or (n,) or (m, n)
    u = p * x + m  # (deg,) or (deg, n) or (deg, m, n)
    v = p * w  # (deg,) or (deg, n) or (deg, m, n)
    fvalues = func(
        u
    )  # (d_1, ..., d_i, deg) or (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
    try:
        _ = np.broadcast_shapes(fvalues.shape[-len(u.shape) :], u.shape)
    except ValueError as err:
        raise ValueError(
            """
            func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
            Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
            """  # noqa: E501
        ) from err

    return np.sum(
        v * fvalues, axis=-v.ndim
    )  # (d_1, ..., d_i) or (d_1, ..., d_i, n) or (d_1, ..., d_i, m, n)


def laguerre_quadrature(
    func: Callable[
        [ST | NumpyST | Array[AtMost2D, NumpyST]],
        np.float64 | Array[AtMost2D, np.float64],
    ],
    a: ST | NumpyST | Array[AtMost2D, NumpyST],
    deg: int = 10,
) -> np.float64 | Array[AtMost2D, np.float64]:
    r"""Numerical integration of :math:`f(x) * exp(-x)` over the interval :math:`[a, \infty]`.

    `func` must accept (deg,), (deg, n) or (deg, m, n) array shapes
    It must handle at least 3 dimensions.
    a can be zero with ndim <= 2.
    """  # noqa: E501
    arr_a = broadcast_bounds(np.asarray(a))  # () or (n,) or (m, n)
    quad = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    x, w = quad
    x = np.expand_dims(
        x, axis=tuple(range(1, arr_a.ndim + 1))
    )  # (deg,), (deg, 1) or (deg, 1, 1)
    w = np.expand_dims(
        w, axis=tuple(range(1, arr_a.ndim + 1))
    )  # (deg,), (deg, 1) or (deg, 1, 1)

    shifted_x = x + arr_a  # (deg,) or (deg, n) or (deg, m, n)
    fvalues = func(
        shifted_x
    )  # (d_1, ..., d_i, deg) or (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
    try:
        _ = np.broadcast_shapes(fvalues.shape[-len(shifted_x.shape) :], shifted_x.shape)
    except ValueError as err:
        # func est une fonction réel univariée et pas multivariée
        raise ValueError(
            """
            func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
            Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
            """  # noqa: E501
        ) from err

    exp_a = np.where(np.exp(-arr_a) == 0, 1.0, np.exp(-arr_a))  # () or (n,) or (m, n)
    return np.sum(
        w * fvalues * exp_a, axis=-shifted_x.ndim
    )  # (d_1, ..., d_i) or (d_1, ..., d_i, n) or (d_1, ..., d_i, m, n)


def unweighted_laguerre_quadrature(
    func: Callable[
        [ST | NumpyST | Array[AtMost2D, NumpyST]],
        np.float64 | Array[AtMost2D, np.float64],
    ],
    a: ST | NumpyST | Array[AtMost2D, NumpyST],
    deg: int = 10,
):
    r"""Numerical integration of :math:`f(x)` over the interval :math:`[a, \infty]`

    `func` must accept (deg,), (deg, n) or (deg, m, n) array shapes
    It must handle at least 3 dimensions.
    a can be zero with ndim <= 2.
    """

    arr_a = broadcast_bounds(np.asarray(a))  # () or (n,) or (m, n)
    quad = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    x, w = quad
    x = np.expand_dims(
        x, axis=tuple(range(1, arr_a.ndim + 1))
    )  # (deg,), (deg, 1) or (deg, 1, 1)
    w = np.expand_dims(
        w, axis=tuple(range(1, arr_a.ndim + 1))
    )  # (deg,), (deg, 1) or (deg, 1, 1)

    shifted_x = x + arr_a  # (deg,) or (deg, n) or (deg, m, n)
    fvalues = func(
        shifted_x
    )  # (d_1, ..., d_i, deg) or (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
    try:
        _ = np.broadcast_shapes(fvalues.shape[-len(shifted_x.shape) :], shifted_x.shape)
    except ValueError as err:
        raise ValueError(
            """
            func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
            Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
            """  # noqa: E501
        ) from err
    return np.sum(
        w * fvalues * np.exp(x), axis=-shifted_x.ndim
    )  # (d_1, ..., d_i) or (d_1, ..., d_i, n) or (d_1, ..., d_i, m, n)
