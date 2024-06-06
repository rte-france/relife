"""
This module defines functions used for numerical integrations

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def gauss_legendre(
    func: Callable,
    lower_bound: FloatArray,
    upper_bound: FloatArray,
    ndim: int = 0,
    deg: int = 100,
) -> FloatArray:
    r"""Gauss-Legendre integration.

    Parameters
    ----------
    func : Callable
        The function to be integrated.
    lower_bound : float or ndarray.
        The lower bound of integration.
    upper_bound : float or 1D array.
        The upper bound of integration.
    ndim : int, optional
        Number of dimensions of the `func`, by default 0.
    deg : int, optional
        Degree of the Legendre polynomials (see numpy.polynomial.legendre.leggauss
        documentation), by default 100.

    Returns
    -------
    ndarray
        The numerical integration of `func` from `lower_bound` to `upper_bound`.

    Notes
    -----
    It is lower_bound numerical integration of:

    .. math::

        \int_a^upper_bound f(x) \mathrm{d} x
    """
    shape = (-1,) + (1,) * ndim
    x, w = np.polynomial.legendre.leggauss(deg)
    x, w = x.reshape(shape), w.reshape(shape)
    p = (upper_bound - lower_bound) / 2
    m = (lower_bound + upper_bound) / 2
    u = p * x + m
    v = p * w
    return np.sum(v * func(u), axis=0)


def quad_laguerre(
    func: Callable, lower_bound: FloatArray, ndim: int = 0, deg: int = 100
) -> FloatArray:
    r"""Numerical integration over the interval `[lower_bound, inf]`.


    Parameters
    ----------
    func : Callable
        The function to be integrated.
    lower_bound : float or ndarray
        The lower bound of integration.
    ndim : int, optional
        Number of dimensions of the `func`, by default 0.
    deg : int, optional
        Degree of the Laguerre polynomials (see numpy.polynomial.laguerre.laggauss
        documentation), by default 100.

    Returns
    -------
    ndarray
        Numerical integration of `func` over the interval `[lower_bound,inf]`.

    Notes
    -----
    The numerical integration using Gauss-Laguerre quadrature of:

    .. math::

        \int_a^{+\infty} f(x) \mathrm{d} x
    """

    return gauss_laguerre(
        lambda x: func(x + lower_bound) * np.exp(x), ndim=ndim, deg=deg
    )


def gauss_laguerre(func: Callable, ndim: int = 0, deg: int = 100) -> FloatArray:
    r"""Gauss-Laguerre integration.

    Parameters
    ----------
    func : Callable
        The function to be integrated.
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
    return np.sum(w * func(x), axis=0)


def shifted_laguerre(
    func: Callable, lower_bound: FloatArray, ndim: int = 0, deg: int = 100
) -> FloatArray:
    r"""Shifted Gauss-Laguerre integration.

    Parameters
    ----------
    func : Callable
        The function to be integrated.
    lower_bound : float or ndarray
        The lower bound of integration.
    ndim : int, optional
        Number of dimensions of the `func`, by default 0.
    deg : int
        Degree of the Laguerre polynomials (see numpy.polynomial.laguerre.laggauss
        documentation), by default 100.

    Returns
    -------
    ndarray
        The Gauss-Laguerre integration of the function `func` from `lower_bound` to infinity.

    Notes
    -----
    It is the numerical integration of:

    .. math::

        \int_a^{+\infty} f(x) e^{-x} \mathrm{d} x
    """

    return gauss_laguerre(
        lambda x: func(x + lower_bound) * np.exp(-lower_bound), ndim=ndim, deg=deg
    )
