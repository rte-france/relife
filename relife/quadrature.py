from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray


"""
(a or b)    out
float       float or (m,)
(m,)        (m,) or (m, n)
(m, n)      (m, n)
"""

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
        if arr_b.ndim <= 1:
            arr_b = arr_b.reshape(-1, 1)
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
    deg: int = 10,
) -> NDArray[np.float64]:
    r"""Numerical integration of func over the interval `[a,b]`

    func must be continuous and expects one input only
    a can be zero
    b must not be inf

    a, b shapes can be either 0d (float like), 1d or 2d
    """

    x, w = np.polynomial.legendre.leggauss(deg)  # (deg,)
    arr_a, arr_b = _reshape_bounds(a, b)  # same shape (m, n)
    if np.any(arr_b == np.inf):
        raise ValueError
    bound_shape = arr_a.shape

    p = (arr_b - arr_a) / 2  # (m, n)
    m = (arr_a + arr_b) / 2  # (m, n)
    u = np.expand_dims(p, -1) * x + np.expand_dims(m, -1) # (m, n, deg)
    v = np.expand_dims(p, -1) * w  # (m, n, deg)
    try:
        #  avoid passing more than 2d to func
        fvalues = func(u.reshape(u.shape[0], -1))  #  (m, n*deg)
        fvalues = fvalues.reshape((fvalues.shape[0], -1, deg))  #  (m, n, deg)
    except ValueError as err:
        func_shape = func(1.0).shape
        if len(func_shape) > 0:
            raise ValueError(
                f"Broadcasting error between a and func. Expected a of shape {(func_shape[0],)} or {(func_shape[0], -1)}"
            )
        raise ValueError from err

    wsum = np.sum(v * fvalues, axis=-1).reshape(bound_shape)
    if max(bound_shape) == 1:
        return np.squeeze(wsum)
    return np.squeeze(wsum.reshape(bound_shape))


def laguerre_quadrature(
    func: Callable[
        [NDArray[np.float64]], NDArray[np.float64]
    ],  # tester avec func : 1d -> 2d / 0d -> 2d / 1d -> 1d , etc.
    a: float | NDArray[np.float64] = 0.0,
    deg: int = 10,
) -> NDArray[np.float64]:
    r"""Numerical integration of `func * exp(-x)` over the interval `[a, inf]`

    func must be continuous and expects one input only
    a can be zero

    a shape can be either 0d (float like), 1d or 2d
    """

    x, w = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    arr_a = _reshape_bounds(a)  # (m, n)
    bound_shape = arr_a.shape

    shifted_x = x + np.expand_dims(arr_a, axis=-1) # (m, n, deg)
    try:
        #  avoid passing more than 2d to func
        fvalues = func(shifted_x.reshape(shifted_x.shape[0], -1))  # (m, n*deg)
        nb_assets = fvalues.shape[0] # m
        fvalues = fvalues.reshape((nb_assets, -1, deg))  # (m, n, deg)
    except ValueError as err:
        func_shape = func(1.0).shape
        if len(func_shape) > 0:
            raise ValueError(
                f"Broadcasting error between a and func. Expected a of shape {(func_shape[0],)} or {(func_shape[0], -1)}"
            )
        raise ValueError from err

    exp_a = np.where(np.exp(-arr_a) == 0, 1.0, np.exp(-arr_a))  # (m, n)
    wsum = np.sum(
        w * fvalues * np.expand_dims(exp_a, axis=-1), axis=-1
    )  # (m, n)

    if max(bound_shape) == 1:
        return np.squeeze(wsum)
    return np.squeeze(wsum.reshape(bound_shape))




def unweighted_laguerre_quadrature(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64] = 0.,
    deg: int = 10,
) -> NDArray[np.float64]:
    r"""Numerical integration of `func` over the interval `[a, inf]`

    func must be continuous and expects one input only
    a can be zero

    a shape can be either 0d (float like), 1d or 2d
    """

    x, w = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    arr_a = _reshape_bounds(a)  # (m, n)
    bound_shape = arr_a.shape

    shifted_x = x + np.expand_dims(arr_a, axis=-1) # (m, n, deg)
    try:
        #  avoid passing more than 2d to func
        fvalues = func(shifted_x.reshape(shifted_x.shape[0], -1))  # (m, n*deg)
        nb_assets = fvalues.shape[0] # m
        fvalues = fvalues.reshape((nb_assets, -1, deg))  # (m, n, deg)
    except ValueError as err:
        func_shape = func(1.0).shape
        if len(func_shape) > 0:
            raise ValueError(
                f"Broadcasting error between a and func. Expected a of shape {(func_shape[0],)} or {(func_shape[0], -1)}"
            )
        raise ValueError from err

    wsum = np.sum(w * fvalues * np.exp(x), axis=-1)  # (m, n)
    if max(bound_shape) == 1:
        return np.squeeze(wsum)
    return np.squeeze(wsum.reshape(bound_shape))

# def quadrature(
#     func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
#     a: float | NDArray[np.float64] = 0.,
#     b: float | NDArray[np.float64] = np.inf,
#     deg: int = 10,
# ):
#     r"""Numerical integration of func over the interval `[a,b]`
#
#     func must be continuous and expects one input only
#     a can be zero
#     b can be inf
#
#     a, b shapes can be either 0d (float like), 1d or 2d
#     """
#     arr_a, arr_b = _reshape_bounds(a, b)
#     if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
#         arr_a = arr_a.flatten()  # (m*n,)
#         arr_b = arr_b.flatten()  # (m*n,)
#
#         integration = np.empty_like(arr_a)  # (m*n,)
#         is_inf = np.isinf(arr_b)
#
#         if arr_a[is_inf].size != 0:
#             integration[is_inf] = unweighted_laguerre_quadrature(
#                 func, arr_a[is_inf].copy(), deg=deg
#             )
#         if arr_a[~is_inf].size != 0:
#             integration[~is_inf] = legendre_quadrature(
#                 func, arr_a[~is_inf].copy(), arr_b[~is_inf].copy(), deg=deg
#             )
#
#         return integration.reshape(
#             np.broadcast_shapes(np.asarray(a).shape, np.asarray(b).shape)
#         )
#     else:  # both are float-like
#         a = arr_a.item()
#         b = arr_b.item()
#         if b == np.inf:
#             return unweighted_laguerre_quadrature(func, a)
#         return legendre_quadrature(func, a, b)
