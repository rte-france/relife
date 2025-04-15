from __future__ import annotations
from typing import Callable, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from relife.lifetime_model import FrozenParametricLifetimeModel


def _reshape_and_broadcast(*args : float | NDArray[np.float64]) -> NDArray[np.float64] | tuple[NDArray[np.float64], ...]:
    def reshape(x : float | NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(x)
        if arr.ndim > 2:
            raise ValueError
        if arr.ndim <= 1:
            arr = arr.reshape(-1, 1)
        if np.any(arr < 0):
            raise ValueError
        return arr
    if len(args) > 1:
        return np.broadcast_arrays(*map(reshape, args))
    return reshape(args[0])


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
    arr_a, arr_b = _reshape_and_broadcast(a, b)  # same shape (m, n)
    if np.any(arr_b == np.inf):
        raise ValueError
    if np.any(arr_a >= arr_b):
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
    arr_a = _reshape_and_broadcast(a)  # (m, n)
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
    arr_a = _reshape_and_broadcast(a)  # (m, n)
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


def ls_integrate(
    model : FrozenParametricLifetimeModel,
    func : Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64] = 0.0,
    b: float | NDArray[np.float64] = np.inf,
    deg: int = 10,
) -> NDArray[np.float64]:
    from relife.lifetime_model import AgeReplacementModel

    def integrand(x: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return func(x) * model.pdf(x)

    match model.baseline:
        case AgeReplacementModel():
            ar = model.args[0].copy()
            arr_a, arr_b, arr_ar = _reshape_and_broadcast(a, b, ar)
            bound_shape = arr_a.shape

            if np.any(arr_a >= arr_b):
                raise ValueError
            arr_b = np.minimum(arr_b, arr_ar)
            is_ar = arr_b == arr_ar

            integration = legendre_quadrature(integrand, arr_a, arr_b, deg=deg)
            if np.any(is_ar):
                integration[is_ar] += func(arr_ar[is_ar].copy()) * model.sf(arr_ar[is_ar].copy())

            if max(bound_shape) == 1:
                return np.squeeze(integration)
            return np.squeeze(integration.reshape(bound_shape))

        case _:
            arr_a, arr_b = _reshape_and_broadcast(a, b)
            if np.any(arr_a >= arr_b):
                raise ValueError
            bound_shape = arr_a.shape

            arr_a = arr_a.flatten()  # (m*n,)
            arr_b = arr_b.flatten()  # (m*n,)

            integration = np.empty_like(arr_a)  # (m*n,)

            is_inf = np.isinf(arr_b)
            arr_b[is_inf] = model.isf(1e-4)

            if np.any(is_inf):
                integration[is_inf] = legendre_quadrature(
                    integrand, arr_a[is_inf].copy(), arr_b[is_inf].copy(), deg=deg
                ) + unweighted_laguerre_quadrature(integrand, b[is_inf].copy(), deg=deg)
            if np.any(~is_inf):
                integration[~is_inf] = legendre_quadrature(
                    integrand, arr_a[~is_inf].copy(), arr_b[~is_inf].copy(), deg=deg
                )

            if max(bound_shape) == 1:
                return np.squeeze(integration)
            return np.squeeze(integration.reshape(bound_shape))