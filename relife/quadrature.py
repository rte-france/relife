from __future__ import annotations
from typing import Callable, TYPE_CHECKING, Optional, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from relife.lifetime_model import ParametricLifetimeModel


def _reshape_and_broadcast_bounds(
    a: float | NDArray[np.float64],
    b: Optional[float | NDArray[np.float64]] = None,
    integrand_nb_assets: int = 1,
):
    """
    nb_assets : int, default 1
        0-axis length of the output returned by the integrand. If it is 1, a or b can be 2d and result is broadcasted.
        If it is greater than 1, a and b must have the same 0-axis length if they are 2d. If the 0-axis has length value
        of 1, it will be broadcasted.
    """

    def reshape(bound: float | NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(bound, dtype=np.float64)
        if np.any(arr < 0):
            raise ValueError("Bound values of the integral can't be lower than 0")
        if arr.ndim > 2:
            raise ValueError("Bound the integral can't have more than 2 dimensions")
        if arr.ndim <= 1:
            arr = np.broadcast_to(arr, (integrand_nb_assets, arr.size))
        if arr.ndim == 2 and integrand_nb_assets != 1:  # maybe bound 0-axis is 1 or m
            if (
                arr.shape[0] != 1 and arr.shape[0] != integrand_nb_assets
            ):  # if 0-axis is m and m != assets -> error
                raise ValueError(
                    f"Invalid bound shape. Got {integrand_nb_assets} nb assets returned by the integrand and a bound of {arr.shape} shape. Bound shape may be {(integrand_nb_assets, arr.shape[1])}."
                )
        return arr

    a = reshape(a)
    if b is not None:
        b = reshape(b)
        try:
            a, b = np.broadcast_arrays(a, b)
            return a.copy(), b.copy()
        except ValueError as err:
            raise ValueError(
                f"Incompatible a, b shapes. Got a.shape, b.shape : {a.shape}, {b.shape}"
            ) from err
    return a


def legendre_quadrature(
    integrand: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64],
    b: float | NDArray[np.float64],
    integrand_nb_assets: int = 1,
    deg: int = 10,
) -> NDArray[np.float64]:
    r"""Numerical integration of func over the interval `[a,b]`

    func must be continuous and expects one input only
    a can be zero
    b must not be inf

    a, b shapes can be either 0d (float like), 1d or 2d
    """

    x, w = np.polynomial.legendre.leggauss(deg)  # (deg,)
    arr_a, arr_b = _reshape_and_broadcast_bounds(
        a, b, integrand_nb_assets=integrand_nb_assets
    )
    if np.any(arr_b == np.inf):
        raise ValueError("Bound values of Legendre quadrature must be finite")
    if np.any(arr_a >= arr_b):
        raise ValueError("Bound values a must be strictly lower than values of b")
    bound_shape = arr_a.shape

    # m == nb_assets
    p = (arr_b - arr_a) / 2  # (m, n)
    m = (arr_a + arr_b) / 2  # (m, n)
    u = np.expand_dims(p, -1) * x + np.expand_dims(m, -1)  # (m, n, deg)
    v = np.expand_dims(p, -1) * w  # (m, n, deg)
    try:
        #  avoid passing more than 2d to func
        fvalues = integrand(u.reshape(u.shape[0], -1))  #  (m, n*deg)
        fvalues = fvalues.reshape((fvalues.shape[0], -1, deg))  #  (m, n, deg)
    except ValueError as err:
        func_shape = integrand(1.0).shape  # (), or (m, 1)
        if len(func_shape) > 0:
            raise ValueError(
                f"Broadcasting error between a and func : func returns {(func_shape[0],)} nb_assets but nb_assets is set to {integrand_nb_assets}"
            )
        raise ValueError from err

    wsum = np.sum(v * fvalues, axis=-1).reshape(bound_shape)
    if integrand_nb_assets == 1:
        return np.squeeze(wsum)
    return wsum


def laguerre_quadrature(
    integrand: Callable[
        [NDArray[np.float64]], NDArray[np.float64]
    ],  # tester avec func : 1d -> 2d / 0d -> 2d / 1d -> 1d , etc.
    a: float | NDArray[np.float64] = 0.0,
    integrand_nb_assets: int = 1,
    deg: int = 10,
) -> NDArray[np.float64]:
    r"""Numerical integration of `func * exp(-x)` over the interval `[a, inf]`

    func must be continuous and expects one input only
    a can be zero

    a shape can be either 0d (float like), 1d or 2d
    """

    x, w = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    # m == nb_assets
    arr_a = _reshape_and_broadcast_bounds(
        a, integrand_nb_assets=integrand_nb_assets
    )  # (m, n)
    bound_shape = arr_a.shape

    shifted_x = x + np.expand_dims(arr_a, axis=-1)  # (m, n, deg)
    try:
        #  avoid passing more than 2d to func
        fvalues = integrand(shifted_x.reshape(shifted_x.shape[0], -1))  # (m, n*deg)
        fvalues = fvalues.reshape((fvalues.shape[0], -1, deg))  #  (m, n, deg)
    except ValueError as err:
        func_shape = integrand(1.0).shape
        if len(func_shape) > 0:
            raise ValueError(
                f"Broadcasting error between a and func : func returns {(func_shape[0],)} nb_assets but nb_assets is set to {integrand_nb_assets}"
            )
        raise ValueError from err

    exp_a = np.where(np.exp(-arr_a) == 0, 1.0, np.exp(-arr_a))  # (m, n)
    wsum = np.sum(w * fvalues * np.expand_dims(exp_a, axis=-1), axis=-1).reshape(
        bound_shape
    )  # (m, n)

    if integrand_nb_assets == 1:
        return np.squeeze(wsum)
    return wsum


def unweighted_laguerre_quadrature(
    integrand: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64] = 0.0,
    integrand_nb_assets: int = 1,
    deg: int = 10,
) -> NDArray[np.float64]:
    r"""Numerical integration of `func` over the interval `[a, inf]`

    func must be continuous and expects one input only
    a can be zero

    a shape can be either 0d (float like), 1d or 2d
    """

    x, w = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    # nb_assets == m
    arr_a = _reshape_and_broadcast_bounds(
        a, integrand_nb_assets=integrand_nb_assets
    )  # (m, n)
    bound_shape = arr_a.shape

    shifted_x = x + np.expand_dims(arr_a, axis=-1)  # (m, n, deg)
    try:
        #  avoid passing more than 2d to func
        fvalues = integrand(shifted_x.reshape(shifted_x.shape[0], -1))  # (m, n*deg)
        fvalues = fvalues.reshape((fvalues.shape[0], -1, deg))  #  (m, n, deg)
    except ValueError as err:
        func_shape = integrand(1.0).shape
        if len(func_shape) > 0:
            raise ValueError(
                f"Broadcasting error between a and func : func returns {(func_shape[0],)} nb_assets but nb_assets is set to {integrand_nb_assets}"
            )
        raise ValueError from err

    wsum = np.sum(w * fvalues * np.exp(x), axis=-1).reshape(bound_shape)  # (m, n)
    if integrand_nb_assets == 1:
        return np.squeeze(wsum)
    return wsum


Args = TypeVarTuple("Args")


# NOTE: ls_integrate is implement here because it depends on _reshape_and_broadcast_bounds that must not be imported elsewhere
def ls_integrate(
    model: ParametricLifetimeModel,
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64] = 0.0,
    b: float | NDArray[np.float64] = np.inf,
    *args: *Args,
    deg: int = 10,
) -> NDArray[np.float64]:
    from relife.lifetime_model import AgeReplacementModel

    frozen_model = model.freeze(*args)

    # IMPORTANT: the nb of assets returned by the integrand is suppposed to be equal the nb assets returned by model.pdf
    # could be check in a future version where ls_integrate would be constructed by an object that carries io shape test
    # on func at the initialization of the constructor
    def integrand(x: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return func(x) * frozen_model.pdf(x)

    match model:
        case AgeReplacementModel():
            nb_assets = frozen_model.nb_assets
            # nb_assets == m
            arr_a, arr_b = _reshape_and_broadcast_bounds(
                a, b, integrand_nb_assets=nb_assets
            )  # (m,n)
            if np.any(arr_a >= arr_b):
                raise ValueError("Bound values a must be strictly lower than values of b")

            arr_ar = frozen_model.args[0].copy()
            try:
                arr_ar, arr_a = np.broadcast_arrays(arr_a, arr_ar)
            except ValueError as err:
                raise ValueError("Incompatible ar shape with given bounds") from err

            bound_shape = arr_a.shape
            arr_b = np.minimum(arr_b, arr_ar)
            is_ar = arr_b == arr_ar

            integration = legendre_quadrature(integrand, arr_a, arr_b, integrand_nb_assets=nb_assets, deg=deg)

            if np.any(is_ar):
                integration[is_ar] += (
                    func(arr_ar[is_ar].copy()) * model.sf(arr_ar[is_ar].copy())
                )

            if nb_assets == 1:
                return np.squeeze(integration)
            return integration.reshape(bound_shape)

        case _:
            nb_assets = frozen_model.nb_assets
            # nb_assets == m
            arr_a, arr_b = _reshape_and_broadcast_bounds(
                a, b, integrand_nb_assets=nb_assets
            )  # (m,n)
            if np.any(arr_a >= arr_b):
                raise ValueError("Bound values a must be strictly lower than values of b")

            bound_shape = arr_a.shape

            is_inf = np.isinf(arr_b)
            arr_b[is_inf] = np.broadcast_to(
                frozen_model.isf(1e-4).reshape(-1, 1), arr_b.shape
            )[
                is_inf
            ]  # frozen_model.isf(1e-4).shape == () or (m,1)

            integration = np.where(
                is_inf,
                legendre_quadrature(
                    integrand, arr_a, arr_b, integrand_nb_assets=nb_assets, deg=deg
                )
                + unweighted_laguerre_quadrature(
                    integrand, arr_b, integrand_nb_assets=nb_assets, deg=deg
                ),
                legendre_quadrature(
                    integrand, arr_a, arr_b, integrand_nb_assets=nb_assets, deg=deg
                ),
            )

            if nb_assets == 1:
                return np.squeeze(integration)
            return integration.reshape(bound_shape)
