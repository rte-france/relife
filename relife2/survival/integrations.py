"""
This module defines functions used for numerical integrations

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray

from relife2.survival.distributions.types import DistributionFunctions
from relife2.survival.regressions.types import RegressionFunctions

FloatArray = NDArray[np.float64]


def ls_integrate(
    func: Callable,
    pf: Union[DistributionFunctions, RegressionFunctions],
    a: np.ndarray,
    b: np.ndarray,
    q0: float = 1e-4,
    ndim: int = 0,
    deg: int = 100,
):
    """
    Args:
        func ():
        pf ():
        a ():
        b ():
        q0 ():
        ndim ():
        deg ():

    Returns:

    """
    b = np.minimum(pf.support_upper_bound, b)

    # pass integrand as an attribute of a ls_integrate object
    def integrand(x):
        return func(x) * pf.pdf(x)

    if np.all(np.isinf(b)):
        b = pf.isf(q0)
        res = quad_laguerre(integrand, b, ndim=ndim, deg=deg)
    else:
        res = 0
    return gauss_legendre(integrand, a, b, ndim=ndim, deg=deg) + res


def gauss_legendre(
    func: Callable,
    a: np.ndarray,
    b: np.ndarray,
    *args: np.ndarray,
    ndim: int = 0,
    deg: int = 100,
) -> np.ndarray:
    r"""Gauss-Legendre integration.

    Parameters
    ----------
    func : Callable
        The function to be integrated.
    a : float or ndarray.
        The lower bound of integration.
    b : float or 1D array.
        The upper bound of integration.
    *args : float or 2D array, optional
        Extra arguments required by the function `func`.
    ndim : int, optional
        Number of dimensions of the `func`, by default 0.
    deg : int, optional
        Degree of the Legendre polynomials (see numpy.polynomial.legendre.leggauss
        documentation), by default 100.

    Returns
    -------
    ndarray
        The numerical integration of `func` from `a` to `b`.

    Notes
    -----
    It is a numerical integration of:

    .. math::

        \int_a^b f(x) \mathrm{d} x
    """
    shape = (-1,) + (1,) * ndim
    x, w = np.polynomial.legendre.leggauss(deg)
    x, w = x.reshape(shape), w.reshape(shape)
    p = (b - a) / 2
    m = (a + b) / 2
    u = p * x + m
    v = p * w
    return np.sum(v * func(u, *args), axis=0)


def quad_laguerre(
    func: Callable, a: np.ndarray, ndim: int = 0, deg: int = 100
) -> np.ndarray:
    r"""Numerical integration over the interval `[a, inf]`.


    Parameters
    ----------
    func : Callable
        The function to be integrated.
    a : float or ndarray
        The lower bound of integration.
    ndim : int, optional
        Number of dimensions of the `func`, by default 0.
    deg : int, optional
        Degree of the Laguerre polynomials (see numpy.polynomial.laguerre.laggauss
        documentation), by default 100.

    Returns
    -------
    ndarray
        Numerical integration of `func` over the interval `[a,inf]`.

    Notes
    -----
    The numerical integration using Gauss-Laguerre quadrature of:

    .. math::

        \int_a^{+\infty} f(x) \mathrm{d} x
    """

    return gauss_laguerre(lambda x: func(x + a) * np.exp(x), ndim=ndim, deg=deg)


def gauss_laguerre(
    func: Callable, *args: FloatArray, ndim: int = 0, deg: int = 100
) -> FloatArray:
    r"""Gauss-Laguerre integration.

    Parameters
    ----------
    func : Callable
        The function to be integrated.
    *args : float or 2D array, optional
        Extra arguments required by the function `func`.
    ndim : int, optional
        Number of dimensions of the `func`, by default 0.
    deg : int
        Degree of the Laguerre polynomials (see numpy.polynomial.laguerre.laggauss
        documentation), by default 100.

    Returns
    -------
    ndarray
        The Gauss-Laguerre integration of the function `func`.

    Notes
    -----
    It is the numerical integration of:

    .. math::

        \int_0^{+\infty} f(x) e^{-x} \mathrm{d} x
    """
    shape = (-1,) + (1,) * ndim
    x, w = np.polynomial.laguerre.laggauss(deg)
    x, w = x.reshape(shape), w.reshape(shape)
    return np.sum(w * func(x, *args), axis=0)


def shifted_laguerre(
    func: Callable, a: FloatArray, ndim: int = 0, deg: int = 100
) -> FloatArray:
    r"""Shifted Gauss-Laguerre integration.

    Parameters
    ----------
    func : Callable
        The function to be integrated.
    a : float or ndarray
        The lower bound of integration.
    ndim : int, optional
        Number of dimensions of the `func`, by default 0.
    deg : int
        Degree of the Laguerre polynomials (see numpy.polynomial.laguerre.laggauss
        documentation), by default 100.

    Returns
    -------
    ndarray
        The Gauss-Laguerre integration of the function `func` from `a` to infinity.

    Notes
    -----
    It is the numerical integration of:

    .. math::

        \int_a^{+\infty} f(x) e^{-x} \mathrm{d} x
    """

    return gauss_laguerre(lambda x: func(x + a) * np.exp(-a), ndim=ndim, deg=deg)
