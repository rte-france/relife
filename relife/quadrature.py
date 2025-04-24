from __future__ import annotations
from typing import Callable, TYPE_CHECKING, Optional, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from relife.lifetime_model import ParametricLifetimeModel


def _check_and_broadcast_bounds(
    a: float | NDArray[np.float64],
    b: Optional[float | NDArray[np.float64]] = None,
):

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
            raise ValueError(
                f"Incompatible a, b shapes. Got a.shape, b.shape : {a.shape}, {b.shape}"
            ) from err
    return a


def legendre_quadrature(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64],
    b: float | NDArray[np.float64],
    deg: int = 10,
) -> NDArray[np.float64]:
    r"""Numerical integration of `func` over the interval `[a,b]`

    func must be continuous and expects one input only
    a can be zero
    b must not be inf

    a, b shapes can be either 0d (float like), 1d or 2d
    """

    x, w = np.polynomial.legendre.leggauss(deg)  # (deg,)
    arr_a, arr_b = _check_and_broadcast_bounds(a, b) # () or (n,) or (m, n)
    if np.any(arr_b == np.inf):
        raise ValueError("Bound values of Legendre quadrature must be finite")
    if np.any(arr_a >= arr_b):
        raise ValueError("Bound values a must be strictly lower than values of b")

    p = (arr_b - arr_a) / 2  # () or (n,) or (m, n)
    m = (arr_a + arr_b) / 2  # () or (n,) or (m, n)
    u = np.expand_dims(p, -1) * x + np.expand_dims(m, -1)  # (deg,) or (n, deg) or (m, n, deg)
    v = np.expand_dims(p, -1) * w  # (deg,) or (n, deg) or (m, n, deg)
    try:
        fvalues = func(u)  # (d_1, ..., d_i, deg) or (d_1, ..., d_i, n, deg) or (d_1, ..., d_i, m, n, deg)
    except ValueError:
        raise ValueError("func must accept input array of 3 dimensions")
    if fvalues.shape[-len(u.shape):] != u.shape:
        raise ValueError(
            f"""
            func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
            Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
            """
        )

    return np.sum(v * fvalues, axis=-1) # (d_1, ..., d_i) or (d_1, ..., d_i, n) or (d_1, ..., d_i, m, n)


def unweighted_laguerre_quadrature(
    func: Callable[[float|NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64] = 0.0,
    deg: int = 10,
) -> NDArray[np.float64]:
    r"""Numerical integration of `func` over the interval `[a, inf]`

    `func` must be continuous and must expect only one input.
    It must handle at least 3 dimensions.
    a can be zero with ndim <= 2.
    """

    x, w = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    arr_a = _check_and_broadcast_bounds(a)  # () or (n,) or (m, n)

    shifted_x = x + np.expand_dims(arr_a, axis=-1)  # (deg,) or (n, deg) or (m, n, deg)

    try:
        fvalues = func(shifted_x)  # (d_1, ..., d_i, deg) or (d_1, ..., d_i, n, deg) or (d_1, ..., d_i, m, n, deg)
    except ValueError:
        raise ValueError("func must accept input array of 3 dimensions")
    if fvalues.shape[-len(shifted_x.shape):] != shifted_x.shape:
        raise ValueError(
            f"""
            func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
            Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
            """
        )
    return np.sum(w * fvalues * np.exp(x), axis=-1) # (d_1, ..., d_i) or (d_1, ..., d_i, n) or (d_1, ..., d_i, m, n)


def laguerre_quadrature(
    func: Callable[[float|NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64] = 0.0,
    deg: int = 10,
) -> NDArray[np.float64]:
    r"""Numerical integration of `func * exp(-x)` over the interval `[a, inf]`

    `func` must be continuous and must expect only one input.
    It must handle at least 3 dimensions.
    a can be zero with ndim <= 2.
    """

    x, w = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    arr_a = _check_and_broadcast_bounds(a)  # () or (n,) or (m, n)

    shifted_x = x + np.expand_dims(arr_a, axis=-1)  # (deg,) or (n, deg) or (m, n, deg)

    try:
        fvalues = func(shifted_x)  # (d_1, ..., d_i, deg) or (d_1, ..., d_i, n, deg) or (d_1, ..., d_i, m, n, deg)
    except ValueError:
        raise ValueError("func must accept input array of 3 dimensions")
    if fvalues.shape[-len(shifted_x.shape):] != shifted_x.shape:
        # func est une fonction réel univariée et pas multivariée
        raise ValueError(
            f"""
            func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
            Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
            """
        )

    exp_a = np.where(np.exp(-arr_a) == 0, 1.0, np.exp(-arr_a))  # (deg,) or (n, deg) or (m, n, deg)
    return np.sum(w * fvalues * exp_a, axis=-1) # (d_1, ..., d_i) or (d_1, ..., d_i, n) or (d_1, ..., d_i, m, n)


Args = TypeVarTuple("Args")

def ls_integrate(
    model: ParametricLifetimeModel[*Args],
    func: Callable[[float|NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64] = 0.0,
    b: float | NDArray[np.float64] = np.inf,
    *args: *Args,
    deg: int = 10,
) -> NDArray[np.float64]:
    """

    Parameters
    ----------
    model
    func
        It must handle at least 2 dimensions.
    a
    b
    args
    deg

    Returns
    -------

    """

    frozen_model = model.freeze(*args)

    def integrand(x: float|NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=np.float64) # (deg,), (n, deg) or (m, n, deg)
        x_shape = x.shape
        x = np.atleast_2d(x) # (n, deg) or (m, n, deg)

        # reshape because model.pdf expects input ndim <= 2
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1) # (m, n*deg)
        # x.shape == (n, deg) or (m, n*deg)
        try:
            fx = func(x) # (d_1, ..., d_i, n, deg) or (d_1, ..., d_i, m, n*deg)
        except ValueError:
            raise ValueError("func must accept input array of 2 dimensions")

        if fx.shape[-len(x.shape):] != x.shape:
            raise ValueError(
                f"""
                func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
                Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
                """
            )

        ushape = fx.shape[:-len(x.shape)]  #   == (d_1, ..., d_i)
        pdf = frozen_model.pdf(x) # (n, deg) or (m, n*deg) because pdf always preserve 2 dim if x is 2 dim
        return (fx * pdf).reshape(ushape + x_shape) # (d_1, ..., d_i, deg) or (d_1, ..., d_i, n, deg) or (d_1, ..., d_i, m, n , deg)

    arr_a, arr_b = _check_and_broadcast_bounds(a, b)  # (m,n)
    if np.any(arr_a >= arr_b):
        raise ValueError("Bound values a must be strictly lower than values of b")
    m, n = arr_a.shape
    if m != 1 and m != frozen_model.nb_assets:
        raise ValueError(f"Incompatible bounds with model. Model nb_assets is {frozen_model.nb_assets}")

    is_inf = np.isinf(arr_b)
    arr_b[is_inf] = np.broadcast_to(
        frozen_model.isf(1e-4).reshape(-1, 1), arr_b.shape
    )[
        is_inf
    ]  # frozen_model.isf(1e-4).shape == () or (m,1)


    integration = np.where(
        is_inf,
        legendre_quadrature(
            integrand, arr_a, arr_b, deg=deg
        )
        + unweighted_laguerre_quadrature(
            integrand, arr_b, deg=deg
        ),
        legendre_quadrature(
            integrand, arr_a, arr_b, deg=deg
        ),
    )
    return integration
