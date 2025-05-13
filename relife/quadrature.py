from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, Optional, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from relife.lifetime_model import ParametricLifetimeModel


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
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
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
    x, w = np.polynomial.legendre.leggauss(deg)  # (deg,)
    x = x.reshape((-1,) + (1,) * arr_a.ndim)  # (deg,), (deg, 1) or (deg, 1, 1)
    w = w.reshape((-1,) + (1,) * arr_a.ndim)  # (deg,), (deg, 1) or (deg, 1, 1)

    if np.any(arr_b == np.inf):
        raise ValueError("Bound values of Legendre quadrature must be finite")
    if np.any(arr_a >= arr_b):
        raise ValueError("Bound values a must be strictly lower than values of b")

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
    func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64] = 0.0,
    deg: int = 10,
) -> np.float64 | NDArray[np.float64]:
    r"""Numerical integration of `func * exp(-x)` over the interval `[a, inf]`

    `func` must accept (deg,), (deg, n) or (deg, m, n) array shapes
    It must handle at least 3 dimensions.
    a can be zero with ndim <= 2.
    """
    arr_a = check_and_broadcast_bounds(a)  # () or (n,) or (m, n)
    x, w = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    x = x.reshape((-1,) + (1,) * arr_a.ndim)  # (deg,), (deg, 1) or (deg, 1, 1)
    w = w.reshape((-1,) + (1,) * arr_a.ndim)  # (deg,), (deg, 1) or (deg, 1, 1)

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
    func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
    a: float | NDArray[np.float64] = 0.0,
    deg: int = 10,
) -> np.float64 | NDArray[np.float64]:
    r"""Numerical integration of `func` over the interval `[a, inf]`

    `func` must accept (deg,), (deg, n) or (deg, m, n) array shapes
    It must handle at least 3 dimensions.
    a can be zero with ndim <= 2.
    """

    x, w = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    arr_a = check_and_broadcast_bounds(a)  # () or (n,) or (m, n)
    x = x.reshape((-1,) + (1,) * arr_a.ndim)  # (deg,), (deg, 1) or (deg, 1, 1)
    w = w.reshape((-1,) + (1,) * arr_a.ndim)  # (deg,), (deg, 1) or (deg, 1, 1)

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


Args = TypeVarTuple("Args")


class LebesgueStieltjesMixin(Generic[*Args]):
    def ls_integrate(
        self: ParametricLifetimeModel[*Args],
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64] = 0.0,
        b: float | NDArray[np.float64] = np.inf,
        *args: *Args,
        deg: int = 10,
    ) -> np.float64 | NDArray[np.float64]:
        r"""
        Lebesgue-Stieltjes integration.

        Parameters
        ----------
        func : callable (in : 1 ndarray , out : 1 ndarray)
            The callable must have only one ndarray object as argument and returns one ndarray object
        a : ndarray (max dim of 2)
            Lower bound(s) of integration.
        b : ndarray (max dim of 2)
            Upper bound(s) of integration. If lower bound(s) is infinite, use np.inf as value.)
        deg : int, default 10
            Degree of the polynomials interpolation

        Returns
        -------
        np.ndarray
            Lebesgue-Stieltjes integral of func from `a` to `b`.
        """

        frozen_model = self.freeze(*args)

        def integrand(x: NDArray[np.float64]) -> NDArray[np.float64]:
            #  x.shape == (deg,), (deg, n) or (deg, m, n)
            # fx : (d_1, ..., d_i, deg), (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
            fx = func(x)
            if fx.shape[-len(x.shape) :] != x.shape:
                raise ValueError(
                    f"""
                    func can't squeeze input dimensions. If x has shape (d_1, ..., d_i), func(x) must have shape (..., d_1, ..., d_i).
                    Ex : if x.shape == (m, n), func(x).shape == (..., m, n).
                    """
                )
            if x.ndim == 3:  # reshape because model.pdf is tested only for input ndim <= 2
                deg, m, n = x.shape
                x = np.rollaxis(x, 1).reshape(m, -1)  # (m, deg*n), roll on m because axis 0 must align with m of args
                pdf = frozen_model.pdf(x)  # (m, deg*n)
                pdf = np.rollaxis(pdf.reshape(m, deg, n), 1, 0)  #  (deg, m, n)
            else:  # ndim == 1 | 2
                # reshape to (1, deg*n) or (1, deg), ie place 1 on axis 0 to allow broadcasting with m of args
                pdf = frozen_model.pdf(x.reshape(1, -1))  # (1, deg*n) or (1, deg)
                pdf = pdf.reshape(x.shape)  # (deg, n) or (deg,)

            # (d_1, ..., d_i, deg) or (d_1, ..., d_i, deg, n) or (d_1, ..., d_i, deg, m, n)
            return fx * pdf

        # if isinstance(model, AgeReplacementModel):
        #     ar, args = frozen_model.args[0], frozen_model.args[1:]
        #     b = np.minimum(ar, b)
        #     w = np.where(b == ar, func(ar) * model.baseline.sf(ar, *args), 0.)

        arr_a, arr_b = check_and_broadcast_bounds(a, b)  # (), (n,) or (m, n)
        if np.any(arr_a >= arr_b):
            raise ValueError("Bound values a must be strictly lower than values of b")
        if arr_a.ndim == 2:
            if arr_a.shape[0] not in (
                1,
                frozen_model.args_nb_assets,
            ) and frozen_model.args_nb_assets not in (1, arr_a.shape[0]):
                raise ValueError(
                    f"Incompatible bounds with model. Model has {frozen_model.nb_assets} nb_assets but a and b have shape {a.shape}, {b.shape}"
                )

        bound_b = frozen_model.isf(1e-4)  #  () or (m, 1), if (m, 1) then arr_b.shape == (m, 1) or (m, n)
        broadcasted_arrs = np.broadcast_arrays(arr_a, arr_b, bound_b)
        arr_a = broadcasted_arrs[0].copy()  # arr_a.shape == arr_b.shape == bound_b.shape
        arr_b = broadcasted_arrs[1].copy()  # arr_a.shape == arr_b.shape == bound_b.shape
        bound_b = broadcasted_arrs[2].copy()  # arr_a.shape == arr_b.shape == bound_b.shape
        is_inf = np.isinf(arr_b)  # () or (n,) or (m, n)
        arr_b = np.where(is_inf, bound_b, arr_b)
        integration = legendre_quadrature(
            integrand, arr_a, arr_b, deg=deg
        )  #  (d_1, ..., d_i), (d_1, ..., d_i, n) or (d_1, ..., d_i, m, n)
        is_inf, _ = np.broadcast_arrays(is_inf, integration)
        return np.where(
            is_inf,
            integration + unweighted_laguerre_quadrature(integrand, arr_b, deg=deg),
            integration,
        )

    def moment(self: ParametricLifetimeModel[*Args], n: int, *args: *Args) -> np.float64 | NDArray[np.float64]:
        """n-th order moment

        Parameters
        ----------
        n : order of the moment, at least 1.
        *args : variadic arguments required by the function

        Returns
        -------
        ndarray of shape (0, )
            n-th order moment.
        """
        if n < 1:
            raise ValueError("order of the moment must be at least 1")
        return self.ls_integrate(
            lambda x: x**n,
            0.0,
            np.inf,
            *args,
            deg=100,
        )  #  high degree of polynome to ensure high precision

    def mean(self: ParametricLifetimeModel[*Args], *args: *Args) -> np.float64 | NDArray[np.float64]:
        return self.moment(1, *args)

    def var(self: ParametricLifetimeModel[*Args], *args: *Args) -> np.float64 | NDArray[np.float64]:
        return self.moment(2, *args) - self.moment(1, *args) ** 2

    def mrl(
        self: ParametricLifetimeModel[*Args], time: float | NDArray[np.float64], *args: *Args
    ) -> np.float64 | NDArray[np.float64]:
        sf = self.sf(time, *args)
        ls = self.ls_integrate(lambda x: x - time, time, np.array(np.inf), *args)
        if sf.ndim < 2:  # 2d to 1d or 0d
            ls = np.squeeze(ls)
        return ls / sf
