from collections.abc import Callable
from typing import TypeAlias, TypeVarTuple

import numpy as np
from optype.numpy import ArrayND

from relife.utils import to_numpy_float64

__all__ = [
    "legendre_quadrature",
    "laguerre_quadrature",
    "unweighted_laguerre_quadrature",
]

ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint
Ts = TypeVarTuple("Ts")


def _control_bounds(*bounds: ST | NumpyST | ArrayND[NumpyST]) -> None:
    for bound in bounds:
        if np.any(bound < 0):
            raise ValueError("Bound values of the integral can't be lower than 0")
    try:
        _ = np.broadcast(*bounds)
    except ValueError as err:
        raise ValueError("Bounds could not be broadcast together") from err


def legendre_quadrature(
    func: Callable[..., np.float64 | ArrayND[np.float64]],
    a: ST | NumpyST | ArrayND[NumpyST],
    b: ST | NumpyST | ArrayND[NumpyST],
    args: tuple[ST | NumpyST | ArrayND[NumpyST], ...] = (),
    deg: int = 10,
) -> np.float64 | ArrayND[np.float64]:
    r"""Numerical integration of :math:`f(x)` over the interval :math:`[a,b]`

    Parameters
    ----------
    func : Callable
        A function of the form `y = func(x, a, b, c, ...)` taking floats or ndarrays
        as inputs and returning a np.float64 or an ndarray. `a, b, c, ...` are extra
        arguments that must be passed in the `args` parameter.
        `(x, a, b, c, ...)` broadcasted shape must be the same than `y`.
    a : float or ndarray
        The lower bound of the integration.
    b : float or ndarray
        The upper bound of the integration. Can't be `np.inf`.
    args : float or ndarray
        Extra arguments used in the function call.
    deg : int, default is 10.
        Number of sample points and weights for the quadrature

    Notes
    -----
    `a`, `b` and extra `args` must be broadcastable.

    Returns
    -------
    out : np.float64 or np.ndarray
        The output shape corresponds to a broadcast between `a`, `b` and `*args`.
    """
    npargs = tuple(to_numpy_float64(arg) for arg in args)
    a = to_numpy_float64(a)  # (*a.shape,)
    b = to_numpy_float64(b)  # (*b.shape,)
    _control_bounds(a, b)
    a, b, *npargs = np.broadcast_arrays(a, b, *npargs)  # (*shape,)
    if np.any(b == np.inf):
        raise ValueError("Bound values of Legendre quadrature must be finite")
    if np.any(a > b):
        raise ValueError("Bound values a must be lower than values of b")
    x, w = np.polynomial.legendre.leggauss(deg)  # (deg,)
    x = np.expand_dims(x, axis=tuple(range(a.ndim)))  # (1, ..., 1, deg)
    w = np.expand_dims(w, axis=tuple(range(a.ndim)))  # (1, ..., 1, deg)
    p = np.expand_dims((b - a) / 2, axis=-1)  # (*shape, 1)
    m = np.expand_dims((a + b) / 2, axis=-1)  # (*shape, 1)
    u = p * x + m  # (*shape, deg)
    v = p * w  # (*shape, deg)
    fvalues = func(u, *npargs)  # (*shape, deg)
    return np.sum(v * fvalues, axis=-1)


def laguerre_quadrature(
    func: Callable[..., np.float64 | ArrayND[np.float64]],
    a: ST | NumpyST | ArrayND[NumpyST],
    args: tuple[ST | NumpyST | ArrayND[NumpyST], ...] = (),
    deg: int = 10,
) -> np.float64 | ArrayND[np.float64]:
    r"""Numerical integration of :math:`f(x) * exp(-x)` over the interval :math:`[a, \infty]`.

     Parameters
     ----------
    func : Callable
        A function of the form `y = func(x, a, b, c, ...)` taking floats or ndarrays
        as inputs and returning a np.float64 or an ndarray. `a, b, c, ...` are extra
        arguments that must be passed in the `args` parameter.
        `(x, a, b, c, ...)` broadcasted shape must be the same than `y`.
    a : float or ndarray
         The lower bound of the integration.
     args : float or ndarray
         Extra arguments used in the function call.
     deg : int, default is 10.
         Number of sample points and weights for the quadrature

     Notes
     -----
     `a` and extra `args` must be broadcastable.

     Returns
     -------
     out : np.float64 or np.ndarray
    """  # noqa: E501
    npargs = tuple(to_numpy_float64(arg) for arg in args)
    a = to_numpy_float64(a)
    a, *npargs = np.broadcast_arrays(a, *npargs)  # shape
    a = np.expand_dims(a, axis=-1)  # (*shape, 1)
    _control_bounds(a)
    x, w = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    x = np.expand_dims(x, axis=tuple(range(a.ndim - 1)))  # (1, ..., 1, deg)
    w = np.expand_dims(w, axis=tuple(range(a.ndim - 1)))  # (1, ..., 1, deg)
    fvalues = func(x + a, *npargs)  # (*shape, deg)
    exp_a = np.where(np.exp(-a) == 0, 1.0, np.exp(-a))  # (*a.shape, 1)
    return np.sum(w * fvalues * exp_a, axis=-1)


def unweighted_laguerre_quadrature(
    func: Callable[..., np.float64 | ArrayND[np.float64]],
    a: ST | NumpyST | ArrayND[NumpyST],
    args: tuple[ST | NumpyST | ArrayND[NumpyST], ...] = (),
    deg: int = 10,
) -> np.float64 | ArrayND[np.float64]:
    r"""Numerical integration of :math:`f(x)` over the interval :math:`[a, \infty]`

    Parameters
    ----------
    func : Callable
        A function of the form `y = func(x, a, b, c, ...)` taking floats or ndarrays
        as inputs and returning a np.float64 or an ndarray. `a, b, c, ...` are extra
        arguments that must be passed in the `args` parameter.
        `(x, a, b, c, ...)` broadcasted shape must be the same than `y`.
    a : float or ndarray
        The lower bound of the integration.
    args : float or ndarray
        Extra arguments used in the function call.
    deg : int, default is 10.
        Number of sample points and weights for the quadrature

    Notes
    -----
    `a` and extra `args` must be broadcastable.

    Returns
    -------
    out : np.float64 or np.ndarray
    """  # noqa: E501

    npargs = tuple(to_numpy_float64(arg) for arg in args)
    a = to_numpy_float64(a)
    a, *npargs = np.broadcast_arrays(a, *npargs)  # shape
    a = np.expand_dims(a, axis=-1)  # (*shape, 1)
    _control_bounds(a)
    x, w = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    x = np.expand_dims(x, axis=tuple(range(a.ndim - 1)))  # (1, ..., 1, deg)
    w = np.expand_dims(w, axis=tuple(range(a.ndim - 1)))  # (1, ..., 1, deg)
    fvalues = func(x + a, *npargs)  # (*shape, deg)
    return np.sum(w * fvalues * np.exp(x), axis=-1)
