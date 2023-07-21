"""Numerical integration, args manipulation, plot, etc."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.special import gamma as gamma_func, digamma as digamma_func, lambertw as lambertw_func

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


def moore_jac_uppergamma_c(P, x, tol=1e-6, print_feedback=False):
    # TODO: critère d'arrêt calculé pour une intégrale voisine, et non cible. Ne devrait pas changer grand chose mais
    #  à modifier
    P_ravel = np.ravel(P).astype(float)
    logic_one = np.logical_and(P_ravel <= x, x <= 1)
    logic_two = x < P_ravel

    series_indices = np.where(np.logical_or(logic_one, logic_two))[0]

    # if((p<=x<=1) | (x<p)): # On this case we use the series expansion of the incomplete gamma
    result = []
    for i in range(len(P_ravel)):
        p = P_ravel[i]
        if i in series_indices:

            # overflow testing
            if np.exp(-x) * (x * np.exp(1) / (p + 1)) ** p < MIN_POSITIVE_FLOAT:
                result.append(0)
            else:

                # Initialization of parameters
                R = x / (1 + p)

                # f = (x ** p) / (p * np.exp(x))
                # d_f = np.exp(-x) * x ** p * (p * np.log(x) - 1) / (p ** 2)
                f = np.exp(-x) * (x ** p) / gamma_func(p + 1)
                d_f = np.exp(-x) * x ** p * (np.log(x) - digamma_func(p + 1)) / gamma_func(p + 1)
                epsilon = tol / (abs(f) + abs(d_f))
                delta = (1 - R) * epsilon / 2

                # determining stopping criteria for the infinite series S and dS:
                n1 = np.ceil(
                    (np.log(epsilon) + np.log(1 - R)) / np.log(R)
                ).astype(int)

                n2 = np.ceil(1 + R / (1 - R)).astype(int)

                if np.log(R) * delta >= -1 / np.exp(1):
                    n3 = np.ceil(np.real(
                        lambertw_func(np.log(R) * delta, k=-1) / np.log(R)
                    )).astype(int)
                    n = max(n1, n2, n3)
                else:
                    n = max(n1, n2)

                # Computing the coefficients C_n and their derivatives

                cn = [1]
                for k in range(1, n + 1):
                    cn.append(x / (p + k) * cn[k - 1])

                harmonic = [1 / (p + k) for k in range(1, n + 1)]
                harmonic.insert(0, 0)
                harmonic = np.cumsum(harmonic)

                cn_derivative = [-cn[k] * harmonic[k] for k in range(1, n + 1)]

                S = sum(cn)
                d_S = sum(cn_derivative)

                if print_feedback:
                    print(f"Series expansion was used. Convergence happened after {n} steps")

                # result.append(gamma_func(p) * digamma_func(p) - S * d_f - f * d_S)
                result.append(S * d_f + f * d_S)

        else:  # On this case we use the continued fraction expansion of the incomplete gamma

            # Parameter initialization
            A = [1, 1 + x]
            B = [x, x * (2 - p + x)]
            d_A = [0, 0]
            d_B = [0, -x]

            # f = np.exp(-x) * x ** p
            # d_f = np.exp(-x) * np.log(x) * x ** p
            f = np.exp(-x) * x ** p / gamma_func(p)
            d_f = np.exp(-x) * x ** p * (np.log(x) - digamma_func(p)) / gamma_func(p)

            S = []
            res = 2 * tol
            k = 2
            while res > tol:

                ak = (k - 1) * (p - k)
                bk = 2 * k - p + x

                A.append(bk * A[k - 1] + ak * A[k - 2])
                B.append(bk * B[k - 1] + ak * B[k - 2])

                d_A.append(bk * d_A[k - 1] - A[k - 1] + ak * d_A[k - 2] + (k - 1) * A[k - 2])
                d_B.append(bk * d_B[k - 1] - B[k - 1] + ak * d_B[k - 2] + (k - 1) * B[k - 2])

                S.append(A[-1] / B[-1])

                if len(S) > 1:
                    res = abs(S[-1] - S[-2]) / S[-1]
                k += 1

            S = S[-1]
            d_S = B[-1] ** (-2) * (B[-1] * d_A[-1] - A[-1] * d_B[-1])

            if print_feedback:
                print(f"Continued fraction expansion was used. Convergence happened after {k - 1} steps")

            # result.append(f * d_S + S * d_f)
            result.append(-f * d_S - S * d_f)

    return np.array(result).reshape(P.shape)
