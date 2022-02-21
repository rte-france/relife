"""Numerical integration, args manipulation, plot, etc."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import Callable, Tuple

MIN_POSITIVE_FLOAT = np.finfo(float).resolution

# numerical integration


def gauss_legendre(
    func: Callable,
    a: np.ndarray,
    b: np.ndarray,
    *args: np.ndarray,
    ndim: int = 0,
    deg: int = 100
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
    func: Callable, a: np.ndarray, *args: np.ndarray, ndim: int = 0, deg: int = 100
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
    f = lambda x, *args: func(x + a, *args) * np.exp(-a)
    return gauss_laguerre(f, *args, ndim=ndim, deg=deg)


def quad_laguerre(
    func: Callable, a: np.ndarray, *args: np.ndarray, ndim: int = 0, deg: int = 100
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
    f = lambda x, *args: func(x + a, *args) * np.exp(x)
    return gauss_laguerre(f, *args, ndim=ndim, deg=deg)


# args manipulation functions


def args_size(*args) -> int:
    """Number of elements on axis=-2.

    Parameters
    ----------
    *args : float or 2D array
        Sequence of arrays.

    Returns
    -------
    int
        The number of elements.

    Raises
    ------
    ValueError
        If `args` have not same size on axis=-2.
    """
    if len(args) == 0:
        return 0
    elif 1 in map(np.ndim, args):
        raise ValueError("args must be float or 2D array")
    else:
        s = set(np.size(arg, axis=-2) if np.ndim(arg) >= 2 else 1 for arg in args)
        if len(s - {1}) <= 1:
            return max(s)
        else:
            raise ValueError("args must have same size on axis=-2")


def args_ndim(*args) -> int:
    """Max number of array dimension in args.

    Parameters
    ----------
    *args : float or ndarray
        Sequence of arrays.

    Returns
    -------
    int
        Maximum number of array dimension in `args`.
    """
    return max(map(np.ndim, args), default=0)


def args_take(indices: np.ndarray, *args) -> Tuple[np.ndarray, ...]:
    """Take elements in each array of args on axis=-2.

    Parameters
    ----------
    indices : ndarray
        The indices of the values to extract for each array.
    *args : float or 2D array
        Sequence of arrays.

    Returns
    -------
    Tuple[ndarray, ...]
        The tuple of arrays where values at indices are extracted.
    """
    return tuple(
        np.take(arg, indices, axis=-2)
        if np.ndim(arg) > 0
        else np.tile(arg, (np.size(indices), 1))
        for arg in args
    )


# plotting


def plot(
    x: np.ndarray,
    y: np.ndarray,
    se: np.ndarray = None,
    alpha_ci: float = 0.05,
    bounds=(-np.inf, np.inf),
    **kwargs
) -> None:
    r"""Plot a function with a confidence interval.

    Parameters
    ----------
    x : 1D array
        x-axis values.
    y : np.ndarray
        y-axis values.
    se : np.ndarray, optional
        The standard error, by default None.
    alpha_ci : float, optional
        :math:`\alpha`-value to define the :math:`100(1-\alpha)\%` confidence
        interval, by default 0.05 corresponding to the 95\% confidence interval.
        If set to None or if se is None, no confidence interval is plotted, by
        default 0.05.
    bounds : tuple, optional
        Bounds for clipping the value of the confidence interval, by default
        (-np.inf, np.inf).
    **kwargs : dict, optional
        Extra arguments to specify the plot properties (see
        matplotlib.pyplot.plot documentation).

    """
    ax = kwargs.pop("ax", plt.gca())
    drawstyle = kwargs.pop("drawstyle", "default")
    (lines,) = ax.plot(x, y, drawstyle=drawstyle, **kwargs)
    if alpha_ci is not None and se is not None:
        z = stats.norm.ppf(1 - alpha_ci / 2)
        yl = np.clip(y - z * se, bounds[0], bounds[1])
        yu = np.clip(y + z * se, bounds[0], bounds[1])
        step = drawstyle.split("-")[1] if "steps-" in drawstyle else None
        ax.fill_between(x, yl, yu, facecolor=lines.get_color(), step=step, alpha=0.25)
    ax.legend()
