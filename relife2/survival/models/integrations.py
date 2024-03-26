"""Numerical integration, args manipulation, plot, etc."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

from typing import Callable

import numpy as np

from .backbone import ProbabilityFunctions

# numerical integration


class LebesgueStieljes:
    def __init__(
        self,
        func: Callable,
        pf: ProbabilityFunctions,
        a: np.ndarray,
        b: np.ndarray,
    ):
        """_summary_

        Args:
            func (Callable): The function to be integrated.
            a (float): lower bound of integration.
            b (float): upper bound of integration.
        """
        self.func = func
        self.pf = pf
        self.a = a
        self.b = b

    def __call__(
        self,
        ndim: int = 0,
        deg: int = 100,
        q0: float = 1e-4,
        *args,
    ) -> np.ndarray:

        if np.isinf(self.b):
            b = self.isf(q0, *args)
            res = quad_laguerre(self.func, b, *args, ndim=ndim, deg=deg)
        else:
            res = 0
        return (
            gauss_legendre(self.func, self.a, b, *args, ndim=ndim, deg=deg)
            + res
        )


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


def gauss_laguerre(
    func: Callable, *args: np.ndarray, ndim: int = 0, deg: int = 100
) -> np.ndarray:
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
    func: Callable,
    a: np.ndarray,
    *args: np.ndarray,
    ndim: int = 0,
    deg: int = 100,
) -> np.ndarray:
    r"""Shifted Gauss-Laguerre integration.

    Parameters
    ----------
    func : Callable
        The function to be integrated.
    a : float or ndarray
        The lower bound of integration.
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
        The Gauss-Laguerre integration of the function `func` from `a` to infinity.

    Notes
    -----
    It is the numerical integration of:

    .. math::

        \int_a^{+\infty} f(x) e^{-x} \mathrm{d} x
    """

    def f(x, *args):
        return func(x + a, *args) * np.exp(-a)

    return gauss_laguerre(f, *args, ndim=ndim, deg=deg)


def quad_laguerre(
    func: Callable,
    a: np.ndarray,
    *args: np.ndarray,
    ndim: int = 0,
    deg: int = 100,
) -> np.ndarray:
    r"""Numerical integration over the interval `[a,inf]`.


    Parameters
    ----------
    func : Callable
        The function to be integrated.
    a : float or ndarray
        The lower bound of integration.
    *args : float or 2D array, optional
        Extra arguments required by the function `func`.
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

    def f(x, *args):
        return func(x + a, *args) * np.exp(x)

    return gauss_laguerre(f, *args, ndim=ndim, deg=deg)
