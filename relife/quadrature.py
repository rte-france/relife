from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray


def _reshape_bounds(
    a: float | NDArray[np.float64],
    b: Optional[float | NDArray[np.float64]] = None,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    arr_a = np.asarray(a)

    if arr_a.ndim > 2:
        raise ValueError
    if arr_a.ndim <= 1:
        arr_a = arr_a.reshape(-1, 1)
    if np.any(arr_a < 0):
        raise ValueError
    if b is not None:
        arr_b = np.asarray(b)
        if arr_b.ndim > 2:
            raise ValueError
        arr_a, arr_b = np.broadcast_arrays(arr_a, arr_b)
        if np.any(arr_a >= arr_b):
            raise ValueError
        return arr_a, arr_b
    else:
        return arr_a


def legendre_quadrature(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64],
    b: float | NDArray[np.float64],
    deg: int = 100,
) -> NDArray[np.float64]:
    r"""Numerical integration of func over the interval `[a,b]`

    func must be continuous and expects one input only
    a can be zero
    b must not be inf

    a, b shapes can be either 0d (float like), 1d or 2d
    """

    arr_a, arr_b = _reshape_bounds(a, b)  # same shape (m, n)
    if np.any(arr_b == np.inf):
        raise ValueError
    x, w = np.polynomial.legendre.leggauss(deg)  # (deg,)
    x, w = x.reshape(-1, 1, 1), w.reshape(-1, 1, 1)  # (deg, 1, 1)
    p = (arr_b - arr_a) / 2  # (m, n)
    m = (arr_a + arr_b) / 2  # (m, n)
    u = (p * x + m).reshape(deg, -1)  # (deg, arr_a.size)
    v = (p * w).reshape(deg, -1)  # (deg, arr_a.size)
    fvalues = func(u)  # (deg, arr_a.size)
    wsum = np.sum(v * fvalues, axis=0)  # (arr_a.size,)
    return wsum.reshape(np.asarray(a).shape)


def laguerre_quadrature(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64],
    deg: int = 100,
) -> NDArray[np.float64]:
    r"""Numerical integration of `func * exp(-x)` over the interval `[a, inf]`

    func must be continuous and expects one input only
    a can be zero

    a shape can be either 0d (float like), 1d or 2d
    """

    arr_a = _reshape_bounds(a)  # (m, n)
    x, w = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    shifted_x = (x.reshape(-1, 1, 1) + arr_a).reshape(deg, -1)  # (deg, arr_a.size)
    fvalues = func(shifted_x)  # (deg, arr_a.size)
    exp_a = np.exp(-arr_a).reshape(1, -1)  # (1, arr_a.size)
    exp_a = np.where(arr_a.reshape(1, -1) == 0, 1.0, exp_a)  # (1, arr_a.size)
    wsum = np.sum(w.reshape(-1, 1) * fvalues * exp_a, axis=0)  # (arr_a.size,)
    return wsum.reshape(np.asarray(a).shape)


def unweighted_laguerre_quadrature(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64],
    deg: int = 100,
) -> NDArray[np.float64]:
    r"""Numerical integration of `func` over the interval `[a, inf]`

    func must be continuous and expects one input only
    a can be zero

    a shape can be either 0d (float like), 1d or 2d
    """

    arr_a = _reshape_bounds(a)  # (m, n) or (n,)
    x, w = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    shifted_x = (x.reshape(-1, 1, 1) + arr_a).reshape(deg, -1)  # (deg, arr_a.size)
    fvalues = func(shifted_x) * np.exp(x.reshape(-1, 1))  # (deg, arr_a.size)
    wsum = np.sum(w.reshape(-1, 1) * fvalues, axis=0)  # (arr_a.size,)
    return wsum.reshape(np.asarray(a).shape)


def quadrature(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64],
    b: Optional[float | NDArray[np.float64]] = None,
    deg: int = 100,
):
    r"""Numerical integration of func over the interval `[a,b]`

    func must be continuous and expects one input only
    a can be zero
    b can be inf

    a, b shapes can be either 0d (float like), 1d or 2d
    """

    if b is not None:
        arr_a, arr_b = _reshape_bounds(a, b)
    else:
        arr_a = _reshape_bounds(a)
        arr_b = np.full_like(arr_a, np.inf)

    arr_a = arr_a.flatten()  # (m*n,)
    arr_b = arr_b.flatten()  # (m*n,)

    integration = np.empty_like(arr_a)  # (m*n,)
    is_inf = np.isinf(arr_b)

    integration[is_inf] = unweighted_laguerre_quadrature(
        func, arr_a[is_inf].copy(), deg=deg
    )
    integration[~is_inf] = legendre_quadrature(
        func, arr_a[~is_inf].copy(), arr_b[~is_inf].copy(), deg=deg
    )

    shape = np.asarray(a).shape
    if np.asarray(b).ndim > len(shape):
        shape = np.asarray(b).shape

    return integration.reshape(shape)
