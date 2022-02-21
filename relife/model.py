"""Generic lifetime models."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import scipy.optimize as optimize
from typing import Callable

from .utils import args_size, args_ndim, gauss_legendre, quad_laguerre


class LifetimeModel(ABC):
    """Generic lifetime model.

    Abstract class for lifetime models, with basic functions pertaining to
    statiscal distributions.
    """

    @abstractmethod
    def support_upper_bound(self, *args: np.ndarray) -> np.ndarray:
        """Support upper bound of the lifetime model.

        Parameters
        ----------
        *args : float or 2D array, optional
            Extra arguments required by the lifetime model.

        Returns
        -------
        float or ndarray
            The support upper bound of the lifetime model with respect to the extra arguments.
        """
        pass

    @abstractmethod
    def sf(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Survival (or reliability) function of the lifetime model.

        Parameters
        ----------
        t : float or 1D array
            Elapsed time.
        *args : float or 2D array, optional
            Extra arguments required by the lifetime model.

        Returns
        -------
        float or ndarray
            The survival function evaluated at `t` with extra arguments `args`.

        Notes
        -----
        If `args` are of type ndarray, the arrays should be broadcastable with :code:`shape[-1]=1`.

        The shape of the result will be:

        .. code-block::

            np.broadcast_shapes(*(np.shape(arg) for arg in args))[:-1] + (np.size(t),)
        """
        pass

    @abstractmethod
    def pdf(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Probability density function.

        Parameters
        ----------
        t : float or 1D array
            Elapsed time.
        *args : float or 2D array, optional
            Extra arguments required by the lifetime model.

        Returns
        -------
        float or ndarray
            The probability density function evaluated at `t` with extra arguments `args`.

        Notes
        -----
        If `args` are of type ndarray, the arrays should be broadcastable with :code:`shape[-1]=1`.

        The shape of the result will be:

        .. code-block::

            np.broadcast_shapes(*(np.shape(arg) for arg in args))[:-1] + (np.size(t),)
        """
        pass

    @abstractmethod
    def isf(self, p: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Inverse survival function.

        Parameters
        ----------
        p : float or 1D array
            Probability.
        *args : float or 2D array, optional
            Extra arguments required by the lifetime model.

        Returns
        -------
        float or ndarray
            Complement quantile corresponding to `p`.

        Notes
        -----
        If `args` are of type ndarray, the arrays should be broadcastable with :code:`shape[-1]=1`.

        The shape of the result will be:

        .. code-block::

            np.broadcast_shapes(*(np.shape(arg) for arg in args))[:-1] + (np.size(p),)
        """
        pass

    @abstractmethod
    def ls_integrate(
        self, func: Callable, a: np.ndarray, b: np.ndarray, *args: np.ndarray, **kwargs
    ) -> np.ndarray:
        r"""Lebesgue-Stieltjes integration.

        The Lebesgue-Stieljes intregration of a function with respect to the
        lifetime model taking into account the probability density function and
        jumps.

        Parameters
        ----------
        func : Callable
            Function or method to integrate on interval [a,b] integrated with
            respect to the lifetime model.
        a : float or 1D array
            Lower bound of integration.
        b : float or 1D array
            Upper bound of integration (use numpy.inf for +infinity).
        *args : float or 2D array, optional
            Extra arguments required by the lifetime model.
        **kwargs: int or float
            Extra keyword arguments required for the Lebesgue-Stieljes
            integration.

        Returns
        -------
        float or ndarray
            Lebesgue-Stieltjes integral of func with respect to `cdf` from `a`
            to `b`.

        Notes
        -----
        The Lebesgue-Stieltjes integral is:

        .. math::

            \int_a^b g(x) \mathrm{d}F(x) = \int_a^b g(x) f(x)\mathrm{d}x +
            \sum_i g(a_i) w_i

        where:

        - :math:`F` is the cumulative distribution function,
        - :math:`f` the probability density function of the lifetime model,
        - :math:`a_i` and :math:`w_i` are the points and weights of the jumps.

        .. [1] Resnick, S. I. (1992). Adventures in stochastic processes.
            Springer Science & Business Media. p176.
        """
        pass

    def cdf(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Cumulative distribution function.

        Parameters
        ----------
        t : float or 1D array
            Elapsed time.
        *args : float or 2D array, optional
            Extra arguments required by the lifetime model.

        Returns
        -------
        float or ndarray
            Cumulative distribution function at `t` with extra `args` .

        Notes
        -----
        If `args` are of type ndarray, the arrays should be broadcastable with :code:`shape[-1]=1`.

        The shape of the result will be:

        .. code-block::

            np.broadcast_shapes(*(np.shape(arg) for arg in args))[:-1] + (np.size(t),)
        """
        return 1 - self.sf(t, *args)

    def rvs(
        self, *args: np.ndarray, size: int = 1, random_state: int = None
    ) -> np.ndarray:
        """Random variable sampling.

        Parameters
        ----------
        *args : float or 2D array, optional
            Extra arguments required by the lifetime model.
        size : int, optional
            Size of sample, by default 1.
        random_state : int, optional
            Random seed, by default None.

        Returns
        -------
        float or ndarray
            Sample of random variates with shape[-1]=size.

        Notes
        -----
        If `args` are of type ndarray, the arrays should be broadcastable with :code:`shape[-1]=1`.

        The shape of the result will be:

        .. code-block::

            np.broadcast_shapes(*(np.shape(arg) for arg in args))[:-1] + (size,)
        """
        if len(args) > 0:
            size = (args_size(*args), size)
        u = np.random.RandomState(seed=random_state).uniform(size=size)
        return self.isf(u, *args)

    def ppf(self, p: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Percent point function.

        The `ppf` is the inverse of `cdf`.

        Parameters
        ----------
        p : float or 1D array
            Probability.
        *args : float or 2D array, optional
            Extra arguments required by the lifetime model.

        Returns
        -------
        float or ndarray
            Quantile corresponding to `p`.

        Notes
        -----
        If `args` are of type ndarray, the arrays should be broadcastable with :code:`shape[-1]=1`.

        The shape of the result will be:

        .. code-block::

            np.broadcast_shapes(*(np.shape(arg) for arg in args))[:-1] + (np.size(p),)
        """
        return self.isf(1 - p, *args)

    def median(self, *args: np.ndarray) -> np.ndarray:
        """Median of the distribution.

        The median is the `ppf` evaluated at 0.5.

        Parameters
        ----------
        *args : float or 2D array, optional
            Extra arguments required by the lifetime model.

        Returns
        -------
        float or ndarray
            The median of the distribution.

        Notes
        -----
        If `args` are of type ndarray, the arrays should be broadcastable with :code:`shape[-1]=1`.

        The shape of the result will be:

        .. code-block::

            np.broadcast_shapes(*(np.shape(arg) for arg in args))[:-1] + (1,)
        """
        return self.ppf(0.5, *args)

    def moment(self, n: int, *args: np.ndarray) -> np.ndarray:
        """N-th order moment of the distribution.

        The n-th order moment is the Lebegue-Stieljes integral of x**n with
        respect to the `cdf`.

        Parameters
        ----------
        n : int, n >=1
            Order of moment.
        *args : float or 2D array, optional
            Extra arguments required by the lifetime model.

        Returns
        -------
        float or ndarray
            N-th order moment of the distribution.

        Notes
        -----
        If `args` are of type ndarray, the arrays should be broadcastable with :code:`shape[-1]=1`.

        The shape of the result will be:

        .. code-block::

            np.broadcast_shapes(*(np.shape(arg) for arg in args))[:-1] + (1,)
        """
        ub = self.support_upper_bound(*args)
        return self.ls_integrate(lambda x: x**n, 0, ub, *args, ndim=args_ndim(*args))

    def mean(self, *args: np.ndarray) -> np.ndarray:
        """Mean of the distribution.

        The mean of a distribution is the moment of the first order.

        Parameters
        ----------
        *args : float or 2D array, optional
            Extra arguments required by the lifetime model.

        Returns
        -------
        float or ndarray
            Mean of the distribution.

        Notes
        -----
        If `args` are of type ndarray, the arrays should be broadcastable with :code:`shape[-1]=1`.

        The shape of the result will be:

        .. code-block::

            np.broadcast_shapes(*(np.shape(arg) for arg in args))[:-1] + (1,)
        """
        return self.moment(1, *args)

    def var(self, *args: np.ndarray) -> np.ndarray:
        """Variance of the distribution.

        Parameters
        ----------
        *args : float or 2D array, optional
            Extra arguments required by the lifetime model.

        Returns
        -------
        float or ndarray
            Variance of the distribution.

        Notes
        -----
        If `args` are of type ndarray, the arrays should be broadcastable with :code:`shape[-1]=1`.

        The shape of the result will be:

        .. code-block::

            np.broadcast_shapes(*(np.shape(arg) for arg in args))[:-1] + (1,)
        """
        return self.moment(2, *args) - self.moment(1, *args) ** 2

    def mrl(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        r"""Mean residual life.

        The mean residual life for an asset aged `t` is the mean of the lifetime
        distribution truncated at `t` on the interval `[t,ub)`.

        Parameters
        ----------
        t : float or 1D array
            Age of the asset

        *args : float or 2D array, optional
            Extra arguments required by the lifetime model.

        Returns
        -------
        float or ndarray
            The mean residual life of assets at age `t`.

        Notes
        -----
        If `args` are of type ndarray, the arrays should be broadcastable with :code:`shape[-1]=1`.

        The shape of the result will be:

        .. code-block::

            np.broadcast_shapes(*(np.shape(arg) for arg in args))[:-1] + (np.size(t),).

        The mean residual life is:

        .. math::

            \mu(t) = \dfrac{\int_t^{\infty} (x - t) \mathrm{d}F(x)}{S(t)}

        where
        :math:`F` is the cumulative distribution function and
        :math:`S` is the survival function.
        """
        ndim = args_ndim(t, *args)
        ub = self.support_upper_bound(*args)
        mask = t >= ub
        if np.any(mask):
            t, ub = np.broadcast_arrays(t, ub)
            t = np.ma.MaskedArray(t, mask)
            ub = np.ma.MaskedArray(ub, mask)
        mu = self.ls_integrate(lambda x: x - t, t, ub, *args, ndim=ndim) / self.sf(
            t, *args
        )
        return np.ma.filled(mu, 0)


@dataclass
class AgeReplacementModel(LifetimeModel):
    r"""Age replacement model.

    Lifetime model where the asset is replaced at age `ar`.

    Notes
    -----
    This is equivalent to the distribution of :math:`\min(X,a_r)` where
    :math:`X` is a baseline lifetime model and ar the age of replacement.
    """

    baseline: LifetimeModel  #: Underlying lifetime model of the asset.

    def support_upper_bound(self, ar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return np.minimum(ar, self.baseline.support_upper_bound(*args))

    def sf(self, t: np.ndarray, ar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return np.where(t < ar, self.baseline.sf(t, *args), 0)

    def pdf(self, t: np.ndarray, ar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return np.where(t < ar, self.baseline.pdf(t, *args), 0)

    def isf(self, p: np.ndarray, ar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return np.minimum(self.baseline.isf(p, *args), ar)

    def ls_integrate(
        self,
        func: Callable,
        a: np.ndarray,
        b: np.ndarray,
        ar: np.ndarray,
        *args: np.ndarray,
        ndim: int = 0,
        deg: int = 100
    ) -> np.ndarray:
        ub = self.support_upper_bound(ar, *args)
        b = np.minimum(ub, b)
        f = lambda x, *args: func(x) * self.baseline.pdf(x, *args)
        w = np.where(b == ar, func(ar) * self.baseline.sf(ar, *args), 0)
        return gauss_legendre(f, a, b, *args, ndim=ndim, deg=deg) + w


class HazardFunctions(ABC):
    """Generic hazard functions.

    Abstract class for the definition of a hazard functions: hazard rate,
    cumulative hazard function, and inverse cumulative hazard function.
    """

    @abstractmethod
    def hf(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Hazard function (or hazard rate).

        The hazard function is the derivative of the cumulative hazard function.

        Parameters
        ----------
        t : float or 1D array
            Elapsed time.
        *args : float or 2D array, optional
            Extra arguments required by the hazard function.

        Returns
        -------
        float or 1D array
            Hazard rate at `t`.

        Notes
        -----
        If `args` are of type ndarray, the arrays should be broadcastable with :code:`shape[-1]=1`.

        The shape of the result will be:

        .. code-block::

            np.broadcast_shapes(*(np.shape(arg) for arg in args))[:-1] + (np.size(t),)
        """
        pass

    @abstractmethod
    def chf(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Cumulative hazard function.

        The cumulative hazard function is the integral of the hazard function.

        Parameters
        ----------
        t : float or 1D array
            Elapsed time.
        *args : float or 2D array, optional
            Extra arguments required by the hazard function.

        Returns
        -------
        float or ndarray
            Cumulative hazard function at `t`.

        Notes
        -----
        If `args` are of type ndarray, the arrays should be broadcastable with :code:`shape[-1]=1`.

        The shape of the result will be:

        .. code-block::

            np.broadcast_shapes(*(np.shape(arg) for arg in args))[:-1] + (np.size(t),)
        """
        pass

    @abstractmethod
    def ichf(self, v: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Inverse cumulative hazard function.

        Parameters
        ----------
        v : float or 1D array
            Cumulative hazard rate
        *args : float or 2D array, optional
            Extra arguments required by the hazard function.

        Returns
        -------
        float or ndarray

        Notes
        -----
        If `args` are of type ndarray, the arrays should be broadcastable with :code:`shape[-1]=1`.

        The shape of the result will be:

        .. code-block::

            np.broadcast_shapes(*(np.shape(arg) for arg in args))[:-1] + (np.size(t),)
        """
        pass


class AbsolutelyContinuousLifetimeModel(LifetimeModel, HazardFunctions):
    """Absolutely continuous lifetime model.

    Abstract class that implements LifetimeModel with HazardFunctions.
    """

    def support_upper_bound(self, *args: np.ndarray) -> float:
        return np.inf

    def sf(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return np.exp(-self.chf(t, *args))

    def pdf(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.hf(t, *args) * self.sf(t, *args)

    def isf(self, p: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.ichf(-np.log(p), *args)

    def ls_integrate(
        self,
        func: Callable,
        a: np.ndarray,
        b: np.ndarray,
        *args: np.ndarray,
        ndim: int = 0,
        deg: int = 100,
        q0: float = 1e-4
    ) -> np.ndarray:
        ub = self.support_upper_bound(*args)
        b = np.minimum(ub, b)
        f = lambda x, *args: func(x) * self.pdf(x, *args)
        if np.all(np.isinf(b)):
            b = self.isf(q0, *args)
            res = quad_laguerre(f, b, *args, ndim=ndim, deg=deg)
        else:
            res = 0
        return gauss_legendre(f, a, b, *args, ndim=ndim, deg=deg) + res


@dataclass
class LeftTruncated(AbsolutelyContinuousLifetimeModel):
    """Left truncation of an absolutely continuous lifetime model.

    Conditional distribution of the lifetime model for an asset having reach age `a0`.
    """

    baseline: AbsolutelyContinuousLifetimeModel  #: Underlying absolutely continuous lifetime model of the asset.

    def chf(self, t: np.ndarray, a0: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.baseline.chf(a0 + t, *args) - self.baseline.chf(a0, *args)

    def hf(self, t: np.ndarray, a0: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.baseline.hf(a0 + t, *args)

    def ichf(self, v: np.ndarray, a0: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.baseline.ichf(v + self.baseline.chf(a0, *args), *args) - a0


@dataclass
class EquilibriumDistribution(AbsolutelyContinuousLifetimeModel):
    """Equilibrium distribution of a lifetime model.

    The equilibirum distribution is the distrbution computed from a lifetime
    model that makes the associated delayed renewal process stationary [1]_.

    .. [1] Ross, S. M. (1996). Stochastic processes. New York: Wiley.
    """

    baseline: LifetimeModel  #: Underlying lifetime model of the asset.

    def support_upper_bound(self, *args: np.ndarray) -> np.ndarray:
        return self.baseline.support_upper_bound(*args)

    def sf(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return 1 - self.cdf(t, *args)

    def cdf(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        ndim = args_ndim(t, *args)
        return gauss_legendre(
            self.baseline.sf, 0, t, *args, ndim=ndim
        ) / self.baseline.mean(*args)

    def pdf(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.baseline.sf(t, *args) / self.baseline.mean(*args)

    def hf(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return 1 / self.baseline.mrl(t, *args)

    def chf(self, t: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return -np.log(self.sf(t, *args))

    def isf(self, p: np.ndarray, *args: np.ndarray) -> np.ndarray:
        f = lambda t, *args: self.sf(t, *args) - p
        t = optimize.newton(f, self.baseline.isf(p, *args), args=args)
        return t

    def ichf(self, v: np.ndarray, *args: np.ndarray) -> np.ndarray:
        return self.isf(np.exp(-v), *args)
