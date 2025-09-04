from abc import ABC

import numpy as np
from scipy.optimize import Bounds, newton
from scipy.special import digamma, exp1, gamma, gammaincc, gammainccinv, polygamma

from relife.data import LifetimeData
from relife.likelihood import LikelihoodFromLifetimes
from relife.quadrature import laguerre_quadrature, legendre_quadrature

from ._base import FittableParametricLifetimeModel, ParametricLifetimeModel
from .regression import LifetimeRegression


class LifetimeDistribution(FittableParametricLifetimeModel, ABC):
    """
    Base class for distribution model.
    """

    def sf(self, time):
        """
        The survival function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return super().sf(time)

    def isf(self, probability):
        """
        The inverse survival function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given probability value(s).
        """
        cumulative_hazard_rate = -np.log(probability + 1e-6)  # avoid division by zero
        return self.ichf(cumulative_hazard_rate)

    def cdf(self, time):
        """
        The cumulative density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return super().cdf(time)

    def pdf(self, time):
        """
        The probability density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return super().pdf(time)

    def ppf(self, probability):
        """
        The percent point function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given probability value(s).
        """
        return super().ppf(probability)

    def moment(self, n):
        """
        n-th order moment

        Parameters
        ----------
        n : int
            order of the moment, at least 1.

        Returns
        -------
        np.float64
        """
        return super().moment(n)

    def median(self):
        """
        The median.

        Returns
        -------
        np.float64
        """
        return self.ppf(0.5)  # no super here to return np.float64

    def jac_sf(self, time, *, asarray=False):
        """
        The jacobian of the survival function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        jac_chf, sf = self.jac_chf(time, asarray=True), self.sf(time)
        jac = -jac_chf * sf
        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_cdf(self, time, *, asarray=False):
        """
        The jacobian of the cumulative density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        jac = -self.jac_sf(time, asarray=True)
        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_pdf(self, time, *, asarray=False):
        """
        The jacobian of the probability density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        jac_hf, hf = self.jac_hf(time, asarray=True), self.hf(time)
        jac_sf, sf = self.jac_sf(time, asarray=True), self.sf(time)
        jac = jac_hf * sf + jac_sf * hf
        if not asarray:
            return np.unstack(jac)
        return jac

    def rvs(self, size, *, nb_assets=None, return_event=False, return_entry=False, seed=None):
        """
        Random variable sampling.

        Parameters
        ----------
        size : int
            Size of the generated sample.
        nb_assets : int, optional
            If nb_assets is not None, 2d arrays of samples are generated.
        return_event : bool, default is False
            If True, returns event indicators along with the sample time values.
        return_entry : bool, default is False
            If True, returns corresponding entry values of the sample time values.
        seed : optional int, np.random.BitGenerator, np.random.Generator, np.random.RandomState, default is None
            If int or BitGenerator, seed for random number generator. If np.random.RandomState or np.random.Generator, use as given.

        Returns
        -------
        float, ndarray or tuple of float or ndarray
            The sample values. If either ``return_event`` or ``return_entry`` is True, returns a tuple containing
            the time values followed by event values, entry values or both.
        """
        return super().rvs(size, nb_assets=nb_assets, return_event=return_event, return_entry=return_entry, seed=seed)

    def ls_integrate(self, func, a, b, deg=10):
        """
        Lebesgue-Stieltjes integration.

        Parameters
        ----------
        func : callable (in : 1 ndarray , out : 1 ndarray)
            The callable must have only one ndarray object as argument and one ndarray object as output
        a : ndarray (maximum number of dimension is 2)
            Lower bound(s) of integration.
        b : ndarray (maximum number of dimension is 2)
            Upper bound(s) of integration. If lower bound(s) is infinite, use np.inf as value.)
        deg : int, default 10
            Degree of the polynomials interpolation

        Returns
        -------
        np.ndarray
            Lebesgue-Stieltjes integral of func from `a` to `b`.
        """
        return super().ls_integrate(func, a, b, deg=deg)

    def _get_initial_params(self, time, event=None, entry=None, departure=None):
        param0 = np.ones(self.nb_params, dtype=np.float64)
        param0[-1] = 1 / np.median(time)
        self.params = param0
        return param0

    def _get_params_bounds(self):
        return Bounds(
            np.full(self.nb_params, np.finfo(float).resolution),
            np.full(self.nb_params, np.inf),
        )

    def fit(self, time, *, event=None, entry=None, departure=None, **options):
        """
        Estimation of the distribution parameters from lifetime data.

        Parameters
        ----------
        time : ndarray (1d or 2d)
            Observed lifetime values.
        event : ndarray of boolean values (1d), default is None
            Boolean indicators tagging lifetime values as right censored or complete.
        entry : ndarray of float (1d), default is None
            Left truncations applied to lifetime values.
        departure : ndarray of float (1d), default is None
            Right truncations applied to lifetime values.
        **options
            Extra arguments used by `scipy.minimize`. Default values are:
                - `method` : `"L-BFGS-B"`
                - `contraints` : `()`
                - `tol` : `None`
                - `callback` : `None`
                - `options` : `None`
                - `bounds` : `self.params_bounds`
                - `x0` : `self.init_params`

        Returns
        -------
        Self
            The current object with the estimated parameters setted inplace.

        Notes
        -----
        Supported lifetime observations format is either 1d-array or 2d-array. 2d-array is more advanced
        format that allows to pass other information as left-censored or interval-censored values. In this case,
        `event` is not needed as 2d-array encodes right-censored values by itself.
        """
        return super().fit(time, event=event, entry=entry, departure=departure, **options)


class Exponential(LifetimeDistribution):
    # noinspection PyUnresolvedReferences
    r"""
    Exponential lifetime distribution.

    The exponential distribution is a 1-parameter distribution with
    :math:`(\lambda)`. The probability density function is:

    .. math::

        f(t) = \lambda e^{-\lambda t}

    where:
        - :math:`\lambda > 0`, the rate parameter,
        - :math:`t\geq 0`, the operating time, age, cycles, etc.

    |

    Parameters
    ----------
    rate : float, optional
        rate parameter

    Attributes
    ----------
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.). If the model is not fitted, the value is None.
    nb_params
    params
    params_names
    plot
    rate
    """

    def __init__(self, rate=None):
        super().__init__(rate=rate)

    @property
    def rate(self):  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("rate")

    def hf(self, time):
        """
        The hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """

        return self.rate * np.ones_like(time)

    def chf(self, time):
        """
        The cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return np.asarray(self.rate) * time

    def mean(self):
        """
        The mean of the distribution.

        Returns
        -------
        np.float64
        """
        return 1 / np.asarray(self.rate)

    def var(self):
        """
        The variance of the distribution.

        Returns
        -------
        np.float64
        """
        return 1 / np.asarray(self.rate) ** 2

    def mrl(self, time):
        """
        The mean residual life function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return 1 / self.rate * np.ones_like(time)

    def ichf(self, cumulative_hazard_rate):
        """
        Inverse cumulative hazard function.

        Parameters
        ----------
        cumulative_hazard_rate : float or np.ndarray
            Cumulative hazard rate value(s) at which to compute the function.
            If ndarray, allowed shapes are (), (n,) or (m, n).

        Returns
        -------
            Function values at each given cumulative hazard rate(s).
        """
        return cumulative_hazard_rate / np.asarray(self.rate)

    def jac_hf(self, time, *, asarray=False):
        """
        The jacobian of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        if isinstance(time, np.ndarray):
            jac = np.expand_dims(np.ones_like(time), axis=0).copy()
        else:
            jac = np.array([1], dtype=np.float64)
        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_chf(self, time, *, asarray=False):
        """
        The jacobian of the cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        if isinstance(time, np.ndarray):
            jac = np.expand_dims(time, axis=0).copy()
        else:
            jac = np.array([time], dtype=np.float64)
        if not asarray:
            return np.unstack(jac)
        return jac

    def dhf(self, time):
        """
        The derivate of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        if isinstance(time, np.ndarray):
            return np.zeros_like(time)
        return np.asarray(0.0)


class Weibull(LifetimeDistribution):
    # noinspection PyUnresolvedReferences
    r"""
    Weibull lifetime distribution.

    The Weibull distribution is a 2-parameter distribution with
    :math:`(c,\lambda)`. The probability density function is:

    .. math::

        f(t) = c \lambda (\lambda t)^{c-1} e^{-(\lambda t)^c}

    where:
        - :math:`c > 0`, the shape parameter,
        - :math:`\lambda > 0`, the rate parameter,
        - :math:`t\geq 0`, the operating time, age, cycles, etc.

    Parameters
    ----------
    shape : float, optional
        shape parameter
    rate : float, optional
        rate parameter

    Attributes
    ----------
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.). If the model is not fitted, the value is None.
    nb_params
    params
    params_names
    plot
    shape
    rate
    """

    def __init__(self, shape=None, rate=None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self):  # optional but better for clarity and type checking
        """Get the current shape value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("shape")

    @property
    def rate(self):  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("rate")

    def hf(self, time):
        """
        The hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.shape * self.rate * (self.rate * np.asarray(time)) ** (self.shape - 1)

    def chf(self, time):
        """
        The cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return (self.rate * np.asarray(time)) ** self.shape

    def mean(self):
        """
        The mean of the distribution.

        Returns
        -------
        np.float64
        """
        return gamma(1 + 1 / self.shape) / self.rate

    def var(self):
        """
        The variance of the distribution.

        Returns
        -------
        np.float64
        """
        return gamma(1 + 2 / self.shape) / self.rate**2 - self.mean() ** 2

    def mrl(self, time):
        """
        The mean residual life function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return (
            gamma(1 / self.shape)
            / (self.rate * self.shape * self.sf(time))
            * gammaincc(
                1 / self.shape,
                (self.rate * time) ** self.shape,
            )
        )

    def ichf(self, cumulative_hazard_rate):
        """
        Inverse cumulative hazard function.

        Parameters
        ----------
        cumulative_hazard_rate : float or np.ndarray
            Cumulative hazard rate value(s) at which to compute the function.
            If ndarray, allowed shapes are (), (n,) or (m, n).

        Returns
        -------
            Function values at each given cumulative hazard rate(s).
        """
        return np.asarray(cumulative_hazard_rate) ** (1 / self.shape) / self.rate

    def jac_hf(self, time, *, asarray=False):
        """
        The jacobian of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        jac = (
            self.rate * (self.rate * time) ** (self.shape - 1) * (1 + self.shape * np.log(self.rate * time)),
            self.shape**2 * (self.rate * time) ** (self.shape - 1),
        )
        if asarray:
            return np.stack(jac)
        return jac

    def jac_chf(self, time, *, asarray=False):
        """
        The jacobian of the cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        jac = (
            np.log(self.rate * time) * (self.rate * time) ** self.shape,
            self.shape * time * (self.rate * time) ** (self.shape - 1),
        )
        if asarray:
            return np.stack(jac)
        return jac

    def dhf(self, time):
        """
        The derivate of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        time = np.asarray(time)
        return self.shape * (self.shape - 1) * self.rate**2 * (self.rate * time) ** (self.shape - 2)


class Gompertz(LifetimeDistribution):
    # noinspection PyUnresolvedReferences
    r"""
    Gompertz lifetime distribution.

    The Gompertz distribution is a 2-parameter distribution with
    :math:`(c,\lambda)`. The probability density function is:

    .. math::

        f(t) = c \lambda e^{\lambda t} e^{ -c \left( e^{\lambda t}-1 \right) }

    where:

        - :math:`c > 0`, the shape parameter,
        - :math:`\lambda > 0`, the rate parameter,
        - :math:`t\geq 0`, the operating time, age, cycles, etc.

    |

    Parameters
    ----------
    shape : float, optional
        shape parameter
    rate : float, optional
        rate parameter

    Attributes
    ----------
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.). If the model is not fitted, the value is None.
    nb_params
    params
    params_names
    plot
    shape
    rate
    """

    def __init__(self, shape=None, rate=None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self):  # optional but better for clarity and type checking
        """Get the current shape value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("shape")

    @property
    def rate(self):  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("rate")

    def hf(self, time):
        """
        The hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.shape * self.rate * np.exp(self.rate * time)

    def chf(self, time):
        """
        The cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.shape * np.expm1(self.rate * time)

    def mean(self):
        """
        The mean of the distribution.

        Returns
        -------
        np.float64
        """
        return np.exp(self.shape) * exp1(self.shape) / self.rate

    def var(self):
        """
        The variance of the distribution.

        Returns
        -------
        np.float64
        """
        return super().var()

    def mrl(self, time):
        """
        The mean residual life function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        z = self.shape * np.exp(self.rate * time)
        return np.exp(z) * exp1(z) / self.rate

    def ichf(self, cumulative_hazard_rate):
        """
        Inverse cumulative hazard function.

        Parameters
        ----------
        cumulative_hazard_rate : float or np.ndarray
            Cumulative hazard rate value(s) at which to compute the function.
            If ndarray, allowed shapes are (), (n,) or (m, n).

        Returns
        -------
            Function values at each given cumulative hazard rate(s).
        """
        return 1 / self.rate * np.log1p(cumulative_hazard_rate / self.shape)

    def jac_hf(self, time, *, asarray=False):
        """
        The jacobian of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        jac = (
            self.rate * np.exp(self.rate * time),
            self.shape * np.exp(self.rate * time) * (1 + self.rate * time),
        )
        if asarray:
            return np.stack(jac)
        return jac

    def jac_chf(self, time, *, asarray=False):
        """
        The jacobian of the cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        jac = (
            np.expm1(self.rate * time),
            self.shape * time * np.exp(self.rate * time),
        )
        if asarray:
            return np.stack(jac)
        return jac

    def dhf(self, time):
        """
        The derivate of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.shape * self.rate**2 * np.exp(self.rate * time)

    def _get_initial_params(self, time, event=None, entry=None, departure=None):
        param0 = np.empty(self.nb_params, dtype=np.float64)
        rate = np.pi / (np.sqrt(6) * np.std(time))
        shape = np.exp(-rate * np.mean(time))
        param0[0] = shape
        param0[1] = rate
        return param0


class Gamma(LifetimeDistribution):
    # noinspection PyUnresolvedReferences
    r"""
    Gamma lifetime distribution.

    The Gamma distribution is a 2-parameter distribution with
    :math:`(c,\lambda)`. The probability density function is:

    .. math::

        f(t) = \frac{\lambda^c t^{c-1} e^{-\lambda t}}{\Gamma(c)}

    where:

        - :math:`c > 0`, the shape parameter,
        - :math:`\lambda > 0`, the rate parameter,
        - :math:`t\geq 0`, the operating time, age, cycles, etc.

    |

    Parameters
    ----------
    shape : float, optional
        shape parameter
    rate : float, optional
        rate parameter

    Attributes
    ----------
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.). If the model is not fitted, the value is None.
    nb_params
    params
    params_names
    plot
    shape
    rate
    """

    def __init__(self, shape=None, rate=None):
        super().__init__(shape=shape, rate=rate)

    def _uppergamma(self, x):
        return gammaincc(self.shape, x) * gamma(self.shape)

    def _jac_uppergamma_shape(self, x):
        return laguerre_quadrature(lambda s: np.log(s) * s ** (self.shape - 1), x, deg=100)

    @property
    def shape(self):  # optional but better for clarity and type checking
        """Get the current shape value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("shape")

    @property
    def rate(self):  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("rate")

    def hf(self, time):
        """
        The hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        x = self.rate * time
        return self.rate * x ** (self.shape - 1) * np.exp(-x) / self._uppergamma(x)

    def chf(self, time):
        """
        The cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        x = self.rate * time
        return np.log(gamma(self.shape)) - np.log(self._uppergamma(x))

    def mean(self):
        """
        The mean of the distribution.

        Returns
        -------
        np.float64
        """
        return np.asarray(self.shape / self.rate)

    def var(self):
        """
        The variance of the distribution.

        Returns
        -------
        np.float64
        """
        return np.asarray(self.shape / (self.rate**2))

    def ichf(self, cumulative_hazard_rate):
        """
        Inverse cumulative hazard function.

        Parameters
        ----------
        cumulative_hazard_rate : float or np.ndarray
            Cumulative hazard rate value(s) at which to compute the function.
            If ndarray, allowed shapes are (), (n,) or (m, n).

        Returns
        -------
            Function values at each given cumulative hazard rate(s).
        """
        return 1 / self.rate * gammainccinv(self.shape, np.exp(-cumulative_hazard_rate))

    def jac_hf(self, time, *, asarray=False):
        """
        The jacobian of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        x = self.rate * time
        y = x ** (self.shape - 1) * np.exp(-x) / self._uppergamma(x) ** 2
        jac = (
            y * ((self.rate * np.log(x) * self._uppergamma(x)) - self.rate * self._jac_uppergamma_shape(x)),
            y * ((self.shape - x) * self._uppergamma(x) + x**self.shape * np.exp(-x)),
        )
        if asarray:
            return np.stack(jac)
        return jac

    def jac_chf(self, time, *, asarray=False):
        """
        The jacobian of the cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        x = self.rate * time
        jac = (
            digamma(self.shape) - self._jac_uppergamma_shape(x) / self._uppergamma(x),
            (x ** (self.shape - 1) * time * np.exp(-x) / self._uppergamma(x)),
        )
        if asarray:
            return np.stack(jac)
        return jac

    def dhf(self, time):
        """
        The derivate of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.hf(time) * ((self.shape - 1) / time - self.rate + self.hf(time))

    def mrl(self, time):
        """
        The mean residual life function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return super().mrl(time)


class LogLogistic(LifetimeDistribution):
    # noinspection PyUnresolvedReferences
    r"""
    Log-logistic probability distribution.

    The Log-logistic distribution is defined as a 2-parameter distribution
    :math:`(c, \lambda)`. The probability density function is:

    .. math::

        f(t) = \frac{c \lambda^c t^{c-1}}{(1+(\lambda t)^{c})^2}

    where:

        - :math:`c > 0`, the shape parameter,
        - :math:`\lambda > 0`, the rate parameter,
        - :math:`t\geq 0`, the operating time, age, cycles, etc.

    |

    Parameters
    ----------
    shape : float, optional
        shape parameter
    rate : float, optional
        rate parameter

    Attributes
    ----------
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.). If the model is not fitted, the value is None.
    nb_params
    params
    params_names
    plot
    shape
    rate
    """

    def __init__(self, shape=None, rate=None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self):  # optional but better for clarity and type checking
        """Get the current shape value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("shape")

    @property
    def rate(self):  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("rate")

    def hf(self, time):
        """
        The hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        x = self.rate * np.asarray(time)
        return self.shape * self.rate * x ** (self.shape - 1) / (1 + x**self.shape)

    def chf(self, time):
        """
        The cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        x = self.rate * time
        return np.log(1 + x**self.shape)

    def mean(self):
        """
        The mean of the distribution.

        Returns
        -------
        np.float64
        """
        b = np.pi / self.shape
        if self.shape <= 1:
            raise ValueError(f"Expectancy only defined for shape > 1: shape = {self.shape}")
        return b / (self.rate * np.sin(b))

    def var(self):
        """
        The variance of the distribution.

        Returns
        -------
        np.float64
        """
        b = np.pi / self.shape
        if self.shape <= 2:
            raise ValueError(f"Variance only defined for shape > 2: shape = {self.shape}")
        return (1 / self.rate**2) * (2 * b / np.sin(2 * b) - b**2 / (np.sin(b) ** 2))

    def ichf(self, cumulative_hazard_rate):
        """
        Inverse cumulative hazard function.

        Parameters
        ----------
        cumulative_hazard_rate : float or np.ndarray
            Cumulative hazard rate value(s) at which to compute the function.
            If ndarray, allowed shapes are (), (n,) or (m, n).

        Returns
        -------
            Function values at each given cumulative hazard rate(s).
        """
        return ((np.exp(cumulative_hazard_rate) - 1) ** (1 / self.shape)) / self.rate

    def jac_hf(self, time, *, asarray=False):
        """
        The jacobian of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        x = self.rate * time
        jac = (
            (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2)
            * (1 + x**self.shape + self.shape * np.log(self.rate * time)),
            (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2) * (self.shape**2 / self.rate),
        )
        if asarray:
            return np.stack(jac)
        return jac

    def jac_chf(self, time, *, asarray=False):
        """
        The jacobian of the cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        x = self.rate * time
        jac = (
            (x**self.shape / (1 + x**self.shape)) * np.log(self.rate * time),
            (x**self.shape / (1 + x**self.shape)) * (self.shape / self.rate),
        )
        if asarray:
            return np.stack(jac)
        return jac

    def dhf(self, time):
        """
        The derivate of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        x = self.rate * np.asarray(time)
        return (
            self.shape
            * self.rate**2
            * x ** (self.shape - 2)
            * (self.shape - 1 - x**self.shape)
            / (1 + x**self.shape) ** 2
        )

    def mrl(self, time):
        return super().mrl(time)


class EquilibriumDistribution(ParametricLifetimeModel):
    r"""Equilibrium distribution.

    The equilibirum distribution is the distrbution computed from a lifetime
    core that makes the associated delayed renewal stochastic_process stationary.

    Parameters
    ----------
    baseline : BaseLifetimeModel
        Underlying lifetime core.

    References
    ----------
    .. [1] Ross, S. M. (1996). Stochastic stochastic_process. New York: Wiley.
    """

    # can't expect baseline to be FrozenParametricLifetimeModel too because it does not have freeze_args
    def __init__(self, baseline):
        super().__init__()
        self.baseline = baseline

    def cdf(self, time, *args):
        return legendre_quadrature(lambda x: self.baseline.sf(x, *args), 0, time) / self.baseline.mean(*args)

    def sf(self, time, *args):
        return 1 - self.cdf(time, *args)

    def pdf(self, time, *args):
        return self.baseline.sf(time, *args) / self.baseline.mean(*args)

    def hf(self, time, *args):
        return 1 / self.baseline.mrl(time, *args)

    def chf(self, time, *args):
        return -np.log(self.sf(time, *args))

    def isf(self, probability, *args):
        return newton(
            lambda x: self.sf(x, *args) - probability,
            self.baseline.isf(probability, *args),
            args=args,
        )

    def ichf(
        self,
        cumulative_hazard_rate,
        *args,
    ):
        return self.isf(np.exp(-cumulative_hazard_rate), *args)


class MinimumDistribution(ParametricLifetimeModel):
    r"""Series structure of n identical and independent components.

    The hazard function of the system is given by:

    .. math::

        h(t) = n \cdot  h_0(t)

    where :math:`h_0` is the baseline hazard function of the components.

    Examples
    --------

    Computing the survival (or reliability) function for 3 structures of 3,6 and
    9 identical and idependent components:

    .. code-block::

        model = MinimumDistribution(Weibull(2, 0.05))
        t = np.arange(0, 10, 0.1)
        n = np.array([3, 6, 9]).reshape(-1, 1)
        model.sf(t, n)
    """

    def __init__(self, baseline):
        super().__init__()
        self.baseline = baseline
        self.fitting_results = None

    def sf(self, time, n, *args):
        return super().sf(time)

    def pdf(self, time, n, *args):
        return super().pdf(time)

    def hf(self, time, n, *args):
        return n * self.baseline.hf(time, *args)

    def chf(self, time, n, *args):
        return n * self.baseline.chf(time, *args)

    def ichf(
        self,
        cumulative_hazard_rate,
        n,
        *args,
    ):
        return self.baseline.ichf(cumulative_hazard_rate / n, *args)

    def dhf(self, time, n, *args):
        return n * self.baseline.dhf(time, *args)

    def jac_chf(self, time, n, *args, asarray=False):
        return n * self.baseline.jac_chf(time, *args, asarray=asarray)

    def jac_hf(self, time, n, *args, asarray=False):
        return n * self.baseline.jac_hf(time, *args, asarray=asarray)

    def jac_sf(self, time, n, *args, asarray=False):
        jac_chf, sf = self.jac_chf(time, n, *args, asarray=True), self.sf(time)
        jac = -jac_chf * sf
        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_cdf(self, time, n, *args, asarray=False):
        jac = -self.jac_sf(time, n, *args, asarray=True)
        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_pdf(self, time, n, *args, asarray=False):
        jac_hf, hf = self.jac_hf(time, n, *args, asarray=True), self.hf(time, n)
        jac_sf, sf = self.jac_sf(time, n, *args, asarray=True), self.sf(time, n)
        jac = jac_hf * sf + jac_sf * hf
        if not asarray:
            return np.unstack(jac)
        return jac

    def ls_integrate(self, func, a, b, n, *args, deg: int = 10):
        return super().ls_integrate(func, a, b, n, *args, deg=deg)

    def fit(
        self,
        time,
        n,
        *args,
        event=None,
        entry=None,
        departure=None,
        **kwargs,
    ):
        # initialize params structure (number of parameters in params tree)
        if isinstance(self.baseline, LifetimeRegression):
            self.baseline._init_params(args[0], *args[1:])
        lifetime_data: LifetimeData = LifetimeData(time, event=event, entry=entry, departure=departure, args=(n, *args))
        likelihood = LikelihoodFromLifetimes(self, lifetime_data)
        fitting_results = likelihood.maximum_likelihood_estimation(**kwargs)
        self.params = fitting_results.optimal_params
        self.fitting_results = fitting_results
        return self
