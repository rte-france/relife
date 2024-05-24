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
