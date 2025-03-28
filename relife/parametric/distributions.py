from abc import abstractmethod, ABC
from typing import Optional, NewType, Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds
from scipy.special import digamma, exp1, gamma, gammaincc, gammainccinv
from typing_extensions import override

from relife.data import LifetimeData
from relife.distributions.mixins import ParametricMixin, LifetimeMixin
from relife.likelihoods.mle import FittingResults, maximum_likelihood_estimation
from relife.quadratures import shifted_laguerre

T = NewType("T", NDArray[np.floating] | NDArray[np.integer] | float | int)


# type is ParametricLifetimeModel[()] or LifetimeModel[()]
class Distribution(ParametricMixin, LifetimeMixin[()], ABC):
    """
    Base class for distribution distributions.
    """

    @override
    def sf(self, time: T) -> NDArray[np.float64]:
        return super().sf(time)

    @override
    def isf(self, probability: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Inverse survival function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate)

    @override
    def cdf(self, time: T) -> NDArray[np.float64]:
        return super().cdf(time)

    def pdf(self, time: T) -> NDArray[np.float64]:
        return super().pdf(time)

    @override
    def ppf(self, probability: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Percent point function.

        The percent point corresponds to the inverse of the cumulative distribution function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return super().ppf(probability)

    @override
    def rvs(self, *, size: int = 1, seed: Optional[int] = None):
        """Random variable sampling.

        Parameters
        ----------
        size : int, default 1
            Size of the sample.
        seed : int, default None
            Random seed.

        Returns
        -------
        np.ndarray
            Sample of random lifetimes.
        """

        return super().rvs(size=size, seed=seed)

    @override
    def moment(self, n: int) -> np.float64:
        """
        n-th order moment of the distribution.

        Parameters
        ----------
        n : int
            Order of the moment, at least 1.

        Returns
        -------
        np.float64
            n-th order moment of the distribution.
        """

        return super().moment(n)

    @override
    def median(self) -> np.float64:
        return super().median()

    def init_params(self, lifetime_data: LifetimeData) -> None:
        param0 = np.ones(self.nb_params)
        param0[-1] = 1 / np.median(lifetime_data.rc.values)
        self.params = param0

    @property
    def params_bounds(self) -> Bounds:
        return Bounds(
            np.full(self.nb_params, np.finfo(float).resolution),
            np.full(self.nb_params, np.inf),
        )

    @abstractmethod
    def jac_hf(
        self,
        time: T,
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def jac_chf(
        self,
        time: T,
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def dhf(
        self,
        time: T,
    ) -> NDArray[np.float64]: ...

    def jac_sf(self, time: T) -> NDArray[np.float64]:
        return -self.jac_chf(time) * self.sf(time)

    def jac_cdf(self, time: T) -> NDArray[np.float64]:
        return -self.jac_sf(time)

    def jac_pdf(self, time: T) -> NDArray[np.float64]:
        return self.jac_hf(time) * self.sf(time) + self.jac_sf(time) * self.hf(time)

    def fit(
        self,
        time: NDArray[np.float64],
        /,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> FittingResults:
        # if update to 3.12 : maximum_likelihood_estimation[()](...), generic functions
        fitting_results = maximum_likelihood_estimation(
            self,
            time,
            event=event,
            entry=entry,
            departure=departure,
            **kwargs,
        )
        self.params = fitting_results.params
        return fitting_results


class Exponential(Distribution):
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
    params : np.ndarray
        The model parameters.
    params_names : np.ndarray
        The model parameters.
    rate : np.float64
        The rate parameter.
    """

    def __init__(self, rate: Optional[float] = None):
        super().__init__()
        self.new_params(rate=rate)

    def hf(self, time: T) -> NDArray[np.float64]:
        return self.rate * np.ones_like(time)

    def chf(self, time: T) -> NDArray[np.float64]:
        return self.rate * time

    @override
    def mean(self) -> np.float64:
        return 1 / self.rate

    @override
    def var(self) -> np.float64:
        return 1 / self.rate**2

    @override
    def mrl(self, time: T) -> NDArray[np.float64]:
        return 1 / self.rate * np.ones_like(time)

    @override
    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return cumulative_hazard_rate / self.rate

    def jac_hf(self, time: T) -> NDArray[np.float64]:
        return np.ones((time.size, 1))

    def jac_chf(self, time: T) -> NDArray[np.float64]:
        return np.ones((time.size, 1)) * time

    def dhf(self, time: T) -> NDArray[np.float64]:
        return np.zeros_like(time)


class Weibull(Distribution):
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
    params : np.ndarray
        The model parameters.
    params_names : np.ndarray
        The model parameters.
    shape : np.float64
        The shape parameter.
    rate : np.float64
        The rate parameter.
    """

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__()
        self.new_params(shape=shape, rate=rate)

    def hf(self, time: T) -> NDArray[np.float64]:
        return self.shape * self.rate * (self.rate * time) ** (self.shape - 1)

    def chf(self, time: T) -> NDArray[np.float64]:
        return (self.rate * time) ** self.shape

    @override
    def mean(self) -> np.float64:
        return gamma(1 + 1 / self.shape) / self.rate

    @override
    def var(self) -> np.float64:
        return gamma(1 + 2 / self.shape) / self.rate**2 - self.mean() ** 2

    @override
    def mrl(self, time: T) -> NDArray[np.float64]:
        return (
            gamma(1 / self.shape)
            / (self.rate * self.shape * self.sf(time))
            * gammaincc(
                1 / self.shape,
                (self.rate * time) ** self.shape,
            )
        )

    @override
    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return cumulative_hazard_rate ** (1 / self.shape) / self.rate

    def jac_hf(self, time: T) -> NDArray[np.float64]:
        return np.column_stack(
            (
                self.rate
                * (self.rate * time) ** (self.shape - 1)
                * (1 + self.shape * np.log(self.rate * time)),
                self.shape**2 * (self.rate * time) ** (self.shape - 1),
            )
        )

    def jac_chf(self, time: T) -> NDArray[np.float64]:
        return np.column_stack(
            (
                np.log(self.rate * time) * (self.rate * time) ** self.shape,
                self.shape * time * (self.rate * time) ** (self.shape - 1),
            )
        )

    def dhf(self, time: T) -> NDArray[np.float64]:
        return (
            self.shape
            * (self.shape - 1)
            * self.rate**2
            * (self.rate * time) ** (self.shape - 2)
        )


class Gompertz(Distribution):
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
    params : np.ndarray
        The model parameters.
    params_names : np.ndarray
        The model parameters.
    shape : np.float64
        The shape parameter.
    rate : np.float64
        The rate parameter.

    """

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__()
        self.new_params(shape=shape, rate=rate)

    def init_params(self, lifetime_data: LifetimeData) -> None:
        param0 = np.empty(self.nb_params, dtype=float)
        rate = np.pi / (np.sqrt(6) * np.std(lifetime_data.rc.values))
        shape = np.exp(-rate * np.mean(lifetime_data.rc.values))
        param0[0] = shape
        param0[1] = rate
        self.params = param0

    def hf(self, time: T) -> NDArray[np.float64]:
        return self.shape * self.rate * np.exp(self.rate * time)

    def chf(self, time: T) -> NDArray[np.float64]:
        return self.shape * np.expm1(self.rate * time)

    @override
    def mean(self) -> np.float64:
        return np.exp(self.shape) * exp1(self.shape) / self.rate

    @override
    def var(self) -> np.float64:
        # return np.array(polygamma(1, 1).item() / self.rate**2)
        pass

    @override
    def mrl(self, time: T) -> NDArray[np.float64]:
        z = self.shape * np.exp(self.rate * time)
        return np.exp(z) * exp1(z) / self.rate

    @override
    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return 1 / self.rate * np.log1p(cumulative_hazard_rate / self.shape)

    def jac_hf(self, time: T) -> NDArray[np.float64]:
        return np.column_stack(
            (
                self.rate * np.exp(self.rate * time),
                self.shape * np.exp(self.rate * time) * (1 + self.rate * time),
            )
        )

    def jac_chf(self, time: T) -> NDArray[np.float64]:
        return np.column_stack(
            (
                np.expm1(self.rate * time),
                self.shape * time * np.exp(self.rate * time),
            )
        )

    def dhf(self, time: T) -> NDArray[np.float64]:
        return self.shape * self.rate**2 * np.exp(self.rate * time)


class Gamma(Distribution):
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
    params : np.ndarray
        The model parameters.
    params_names : np.ndarray
        The model parameters.
    shape : np.float64
        The shape parameter.
    rate : np.float64
        The rate parameter.

    """

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__()
        self.new_params(shape=shape, rate=rate)

    def _uppergamma(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return gammaincc(self.shape, x) * gamma(self.shape)

    def _jac_uppergamma_shape(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return shifted_laguerre(
            lambda s: np.log(s) * s ** (self.shape - 1),
            x,
            ndim=np.ndim(x),
        )

    def hf(self, time: T) -> NDArray[np.float64]:
        x = self.rate * time
        return self.rate * x ** (self.shape - 1) * np.exp(-x) / self._uppergamma(x)

    def chf(self, time: T) -> NDArray[np.float64]:
        x = self.rate * time
        return np.log(gamma(self.shape)) - np.log(self._uppergamma(x))

    @override
    def mean(self) -> np.float64:
        return np.array(self.shape / self.rate)

    @override
    def var(self) -> np.float64:
        return self.shape / (self.rate**2)

    @override
    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return 1 / self.rate * gammainccinv(self.shape, np.exp(-cumulative_hazard_rate))

    def jac_hf(self, time: T) -> NDArray[np.float64]:
        x = self.rate * time
        return (
            x ** (self.shape - 1)
            * np.exp(-x)
            / self._uppergamma(x) ** 2
            * np.column_stack(
                (
                    self.rate * np.log(x) * self._uppergamma(x)
                    - self.rate * self._jac_uppergamma_shape(x),
                    (self.shape - x) * self._uppergamma(x) + x**self.shape * np.exp(-x),
                )
            )
        )

    def jac_chf(
        self,
        time: T,
    ) -> NDArray[np.float64]:
        x = self.rate * time
        return np.column_stack(
            (
                digamma(self.shape)
                - self._jac_uppergamma_shape(x) / self._uppergamma(x),
                x ** (self.shape - 1) * time * np.exp(-x) / self._uppergamma(x),
            )
        )

    def dhf(self, time: T) -> NDArray[np.float64]:
        return self.hf(time) * ((self.shape - 1) / time - self.rate + self.hf(time))

    @override
    def mrl(self, time: T) -> NDArray[np.float64]:
        return super().mrl(time)


class LogLogistic(Distribution):
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
    params : np.ndarray
        The model parameters.
    params_names : np.ndarray
        The model parameters.
    shape : np.float64
        The shape parameter.
    rate : np.float64
        The rate parameter.

    """

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__()
        self.new_params(shape=shape, rate=rate)

    def hf(self, time: T) -> NDArray[np.float64]:
        x = self.rate * time
        return self.shape * self.rate * x ** (self.shape - 1) / (1 + x**self.shape)

    def chf(self, time: T) -> NDArray[np.float64]:
        x = self.rate * time
        return np.array(np.log(1 + x**self.shape))

    @override
    def mean(self) -> NDArray[np.float64]:
        b = np.pi / self.shape
        if self.shape <= 1:
            raise ValueError(
                f"Expectancy only defined for shape > 1: shape = {self.shape}"
            )
        return b / (self.rate * np.sin(b))

    @override
    def var(self) -> np.float64:
        b = np.pi / self.shape
        if self.shape <= 2:
            raise ValueError(
                f"Variance only defined for shape > 2: shape = {self.shape}"
            )
        return (1 / self.rate**2) * (2 * b / np.sin(2 * b) - b**2 / (np.sin(b) ** 2))

    @override
    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return ((np.exp(cumulative_hazard_rate) - 1) ** (1 / self.shape)) / self.rate

    def jac_hf(self, time: T) -> NDArray[np.float64]:
        x = self.rate * time
        return np.column_stack(
            (
                (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2)
                * (1 + x**self.shape + self.shape * np.log(self.rate * time)),
                (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2)
                * (self.shape**2 / self.rate),
            )
        )

    def jac_chf(self, time: T) -> NDArray[np.float64]:
        x = self.rate * time
        return np.column_stack(
            (
                (x**self.shape / (1 + x**self.shape)) * np.log(self.rate * time),
                (x**self.shape / (1 + x**self.shape)) * (self.shape / self.rate),
            )
        )

    def dhf(self, time: T) -> NDArray[np.float64]:
        x = self.rate * time
        return (
            self.shape
            * self.rate**2
            * x ** (self.shape - 2)
            * (self.shape - 1 - x**self.shape)
            / (1 + x**self.shape) ** 2
        )

    @override
    def mrl(self, time: T) -> NDArray[np.float64]:
        return super().mrl(time)


TIME_BASE_DOCSTRING = """
{name}.

Parameters
----------
time : float or np.ndarray
    Elapsed time value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.

Returns
-------
np.float64 or np.ndarray
    Function values at each given time(s).
"""


ICHF_DOCSTRING = """
Inverse cumulative hazard function.

Parameters
----------
cumulative_hazard_rate : float or np.ndarray
    Cumulative hazard rate value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.

Returns
-------
np.float64 or np.ndarray
    Function values at each given time(s).
"""

MOMENT_BASE_DOCSTRING = """
{name}.

Returns
-------
np.float64
    {name} value.
"""


for class_obj in (Exponential, Weibull, Gompertz, Gamma, LogLogistic):
    class_obj.sf.__doc__ = TIME_BASE_DOCSTRING.format(name="The survival function")
    class_obj.hf.__doc__ = TIME_BASE_DOCSTRING.format(name="The hazard function")
    class_obj.chf.__doc__ = TIME_BASE_DOCSTRING.format(
        name="The cumulative hazard function"
    )
    class_obj.pdf.__doc__ = TIME_BASE_DOCSTRING.format(
        name="The probability density function"
    )
    class_obj.mrl.__doc__ = TIME_BASE_DOCSTRING.format(
        name="The mean residual life function"
    )
    class_obj.cdf.__doc__ = TIME_BASE_DOCSTRING.format(
        name="The cumulative distribution function"
    )
    class_obj.dhf.__doc__ = TIME_BASE_DOCSTRING.format(
        name="The derivative of the hazard function"
    )
    class_obj.jac_hf.__doc__ = TIME_BASE_DOCSTRING.format(
        name="The jacobian of the hazard function"
    )
    class_obj.jac_chf.__doc__ = TIME_BASE_DOCSTRING.format(
        name="The jacobian of the cumulative hazard function"
    )
    class_obj.jac_sf.__doc__ = TIME_BASE_DOCSTRING.format(
        name="The jacobian of the survival function"
    )
    class_obj.jac_pdf.__doc__ = TIME_BASE_DOCSTRING.format(
        name="The jacobian of the probability density function"
    )
    class_obj.jac_cdf.__doc__ = TIME_BASE_DOCSTRING.format(
        name="The jacobian of the cumulative distribution function"
    )

    class_obj.mean.__doc__ = MOMENT_BASE_DOCSTRING.format(name="The mean")
    class_obj.var.__doc__ = MOMENT_BASE_DOCSTRING.format(name="The variance")
    class_obj.median.__doc__ = MOMENT_BASE_DOCSTRING.format(name="The median")

    class_obj.ichf.__doc__ = ICHF_DOCSTRING
