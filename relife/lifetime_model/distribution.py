from typing import Literal, Optional, TypeVarTuple, overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton
from scipy.special import digamma, exp1, gamma, gammaincc, gammainccinv, polygamma
from typing_extensions import override

from relife.lifetime_model import LifetimeRegression
from relife.quadrature import laguerre_quadrature, legendre_quadrature

from ._base import LifetimeDistribution, ParametricLifetimeModel


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

    def __init__(self, rate: Optional[float] = None):
        super().__init__(rate=rate)

    @property
    def rate(self) -> float:  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("rate")

    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.rate * np.ones_like(time)

    def chf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return np.asarray(self.rate) * time

    @override
    def mean(self) -> np.float64:
        return 1 / np.asarray(self.rate)

    @override
    def var(self) -> np.float64:
        return 1 / np.asarray(self.rate) ** 2

    @override
    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return 1 / self.rate * np.ones_like(time)

    @override
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return cumulative_hazard_rate / np.asarray(self.rate)

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False],
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True],
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_hf(self, time: float | NDArray[np.float64], *, asarray: bool = False) -> np.float64 | NDArray[np.float64]:
        if isinstance(time, np.ndarray):
            jac = np.expand_dims(np.ones_like(time), axis=0).copy()
        else:
            jac = np.array([1], dtype=np.float64)
        if not asarray:
            return np.unstack(jac)
        return jac

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False],
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True],
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: bool = False
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:
        if isinstance(time, np.ndarray):
            jac = np.expand_dims(time, axis=0).copy()
        else:
            jac = np.array([time], dtype=np.float64)
        if not asarray:
            return np.unstack(jac)
        return jac

    def dhf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
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

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self) -> float:  # optional but better for clarity and type checking
        """Get the current shape value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("shape")

    @property
    def rate(self) -> float:  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("rate")

    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.shape * self.rate * (self.rate * np.asarray(time)) ** (self.shape - 1)

    def chf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return (self.rate * np.asarray(time)) ** self.shape

    @override
    def mean(self) -> np.float64:
        return gamma(1 + 1 / self.shape) / self.rate

    @override
    def var(self) -> np.float64:
        return gamma(1 + 2 / self.shape) / self.rate**2 - self.mean() ** 2

    @override
    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return (
            gamma(1 / self.shape)
            / (self.rate * self.shape * self.sf(time))
            * gammaincc(
                1 / self.shape,
                (self.rate * time) ** self.shape,
            )
        )

    @override
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return np.asarray(cumulative_hazard_rate) ** (1 / self.shape) / self.rate

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False],
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True],
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: bool = False
    ) -> tuple[np.float64, np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        jac = (
            self.rate * (self.rate * time) ** (self.shape - 1) * (1 + self.shape * np.log(self.rate * time)),
            self.shape**2 * (self.rate * time) ** (self.shape - 1),
        )
        if asarray:
            return np.stack(jac)
        return jac

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False],
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True],
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: bool = False
    ) -> tuple[np.float64, np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        jac = (
            np.log(self.rate * time) * (self.rate * time) ** self.shape,
            self.shape * time * (self.rate * time) ** (self.shape - 1),
        )
        if asarray:
            return np.stack(jac)
        return jac

    def dhf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
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

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self) -> float:  # optional but better for clarity and type checking
        """Get the current shape value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("shape")

    @property
    def rate(self) -> float:  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("rate")

    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.shape * self.rate * np.exp(self.rate * time)

    def chf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.shape * np.expm1(self.rate * time)

    @override
    def mean(self) -> np.float64:
        return np.exp(self.shape) * exp1(self.shape) / self.rate

    @override
    def var(self) -> np.float64:
        return polygamma(1, 1) / self.rate**2

    @override
    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        z = self.shape * np.exp(self.rate * time)
        return np.exp(z) * exp1(z) / self.rate

    @override
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return 1 / self.rate * np.log1p(cumulative_hazard_rate / self.shape)

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False],
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True],
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: bool = False
    ) -> tuple[np.float64, np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        jac = (
            self.rate * np.exp(self.rate * time),
            self.shape * np.exp(self.rate * time) * (1 + self.rate * time),
        )
        if asarray:
            return np.stack(jac)
        return jac

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False],
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True],
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: bool = False
    ) -> tuple[np.float64, np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        jac = (
            np.expm1(self.rate * time),
            self.shape * time * np.exp(self.rate * time),
        )
        if asarray:
            return np.stack(jac)
        return jac

    def dhf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.shape * self.rate**2 * np.exp(self.rate * time)


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

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self) -> float:  # optional but better for clarity and type checking
        """Get the current shape value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("shape")

    @property
    def rate(self) -> float:  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("rate")

    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        x = self.rate * time
        return self.rate * x ** (self.shape - 1) * np.exp(-x) / self._uppergamma(x)

    def chf(self, time: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        x = self.rate * time
        return np.log(gamma(self.shape)) - np.log(self._uppergamma(x))

    @override
    def mean(self) -> np.float64:
        return np.asarray(self.shape / self.rate)

    @override
    def var(self) -> np.float64:
        return np.asarray(self.shape / (self.rate**2))

    @override
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return 1 / self.rate * gammainccinv(self.shape, np.exp(-cumulative_hazard_rate))

    def _uppergamma(self, x: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return gammaincc(self.shape, x) * gamma(self.shape)

    def _jac_uppergamma_shape(self, x: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return laguerre_quadrature(lambda s: np.log(s) * s ** (self.shape - 1), x, deg=100)

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False],
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True],
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: bool = False
    ) -> tuple[np.float64, np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = self.rate * time
        y = x ** (self.shape - 1) * np.exp(-x) / self._uppergamma(x) ** 2
        jac = (
            y * ((self.rate * np.log(x) * self._uppergamma(x)) - self.rate * self._jac_uppergamma_shape(x)),
            y * ((self.shape - x) * self._uppergamma(x) + x**self.shape * np.exp(-x)),
        )
        if asarray:
            return np.stack(jac)
        return jac

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False],
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True],
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: bool = False
    ) -> tuple[np.float64, np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = self.rate * time
        jac = (
            digamma(self.shape) - self._jac_uppergamma_shape(x) / self._uppergamma(x),
            (x ** (self.shape - 1) * time * np.exp(-x) / self._uppergamma(x)),
        )
        if asarray:
            return np.stack(jac)
        return jac

    def dhf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return self.hf(time) * ((self.shape - 1) / time - self.rate + self.hf(time))

    @override
    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
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

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self) -> float:  # optional but better for clarity and type checking
        """Get the current shape value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("shape")

    @property
    def rate(self) -> float:  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params.get_param_value("rate")

    def hf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        x = self.rate * np.asarray(time)
        return self.shape * self.rate * x ** (self.shape - 1) / (1 + x**self.shape)

    def chf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        x = self.rate * time
        return np.log(1 + x**self.shape)

    @override
    def mean(self) -> np.float64:
        b = np.pi / self.shape
        if self.shape <= 1:
            raise ValueError(f"Expectancy only defined for shape > 1: shape = {self.shape}")
        return b / (self.rate * np.sin(b))

    @override
    def var(self) -> np.float64:
        b = np.pi / self.shape
        if self.shape <= 2:
            raise ValueError(f"Variance only defined for shape > 2: shape = {self.shape}")
        return (1 / self.rate**2) * (2 * b / np.sin(2 * b) - b**2 / (np.sin(b) ** 2))

    @override
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return ((np.exp(cumulative_hazard_rate) - 1) ** (1 / self.shape)) / self.rate

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False],
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True],
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_hf(
        self, time: float | NDArray[np.float64], *, asarray: bool = False
    ) -> tuple[np.float64, np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = self.rate * time
        jac = (
            (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2)
            * (1 + x**self.shape + self.shape * np.log(self.rate * time)),
            (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2) * (self.shape**2 / self.rate),
        )
        if asarray:
            return np.stack(jac)
        return jac

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[False],
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        *,
        asarray: Literal[True],
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_chf(
        self, time: float | NDArray[np.float64], *, asarray: bool = False
    ) -> tuple[np.float64, np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = self.rate * time
        jac = (
            (x**self.shape / (1 + x**self.shape)) * np.log(self.rate * time),
            (x**self.shape / (1 + x**self.shape)) * (self.shape / self.rate),
        )
        if asarray:
            return np.stack(jac)
        return jac

    def dhf(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        x = self.rate * np.asarray(time)
        return (
            self.shape
            * self.rate**2
            * x ** (self.shape - 2)
            * (self.shape - 1 - x**self.shape)
            / (1 + x**self.shape) ** 2
        )

    @override
    def mrl(self, time: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
        return super().mrl(time)


Args = TypeVarTuple("Args")


class EquilibriumDistribution(ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]):
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
    def __init__(self, baseline: ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]):
        super().__init__()
        self.baseline = baseline

    @property
    def args_names(self) -> tuple[str, ...]:
        return self.baseline.args_names

    @override
    def cdf(
        self, time: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        return legendre_quadrature(lambda x: self.baseline.sf(x, *args), 0, time) / self.baseline.mean(*args)

    def sf(
        self, time: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        return 1 - self.cdf(time, *args)

    def pdf(
        self, time: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        return self.baseline.sf(time, *args) / self.baseline.mean(*args)

    def hf(
        self, time: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        return 1 / self.baseline.mrl(time, *args)

    def chf(
        self, time: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        return -np.log(self.sf(time, *args))

    @override
    def isf(
        self, probability: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        return newton(
            lambda x: self.sf(x, *args) - probability,
            self.baseline.isf(probability, *args),
            args=args,
        )

    @override
    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        return self.isf(np.exp(-cumulative_hazard_rate), *args)


class MinimumDistribution(ParametricLifetimeModel[*tuple[float | NDArray[np.float64], ...]]):
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

    def sf(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]:
        pass

    def hf(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]:
        pass

    def chf(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]:
        pass

    def pdf(self, time: float | NDArray[np.float64], *args: *Args) -> np.float64 | NDArray[np.float64]:
        pass

    def __init__(self, baseline: LifetimeDistribution | LifetimeRegression):
        super().__init__()
        self.baseline = baseline

    # def _chf(
    #     self, params: np.ndarray, t: np.ndarray, n: np.ndarray, *args: np.ndarray
    # ) -> np.ndarray:
    #     return n * self.baseline._chf(params, t, *args)
    #
    # def _hf(
    #     self, params: np.ndarray, t: np.ndarray, n: np.ndarray, *args: np.ndarray
    # ) -> np.ndarray:
    #     return n * self.baseline._hf(params, t, *args)
    #
    # def _dhf(
    #     self, params: np.ndarray, t: np.ndarray, n: np.ndarray, *args: np.ndarray
    # ) -> np.ndarray:
    #     return n * self.baseline._dhf(params, t, *args)
    #
    # def _jac_chf(
    #     self, params: np.ndarray, t: np.ndarray, n: np.ndarray, *args: np.ndarray
    # ) -> np.ndarray:
    #     return n * self.baseline._jac_chf(params, t, *args)
    #
    # def _jac_hf(
    #     self, params: np.ndarray, t: np.ndarray, n: np.ndarray, *args: np.ndarray
    # ) -> np.ndarray:
    #     return n * self.baseline._jac_hf(params, t, *args)
    #
    # def _ichf(
    #     self, params: np.ndarray, v: np.ndarray, n: np.ndarray, *args: np.ndarray
    # ) -> np.ndarray:
    #     return self.baseline._ichf(params, v / n, *args)


TIME_BASE_DOCSTRING = """
{name}.

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


JAC_BASE_DOCSTRING = """
{name}.

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


MOMENT_BASE_DOCSTRING = """
{name}.

Returns
-------
np.float64
    {name} value.
"""


PROBABILITY_BASE_DOCSTRING = """
{name}.

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


for class_obj in (Exponential, Weibull, Gompertz, Gamma, LogLogistic):
    class_obj.sf.__doc__ = TIME_BASE_DOCSTRING.format(name="The survival function")
    class_obj.hf.__doc__ = TIME_BASE_DOCSTRING.format(name="The hazard function")
    class_obj.chf.__doc__ = TIME_BASE_DOCSTRING.format(name="The cumulative hazard function")
    class_obj.pdf.__doc__ = TIME_BASE_DOCSTRING.format(name="The probability density function")
    class_obj.cdf.__doc__ = TIME_BASE_DOCSTRING.format(name="The cumulative distribution function")
    class_obj.mrl.__doc__ = TIME_BASE_DOCSTRING.format(name="The mean residual life function")
    class_obj.dhf.__doc__ = TIME_BASE_DOCSTRING.format(name="The derivative of the hazard function")
    class_obj.jac_hf.__doc__ = JAC_BASE_DOCSTRING.format(name="The jacobian of the hazard function")
    class_obj.jac_chf.__doc__ = JAC_BASE_DOCSTRING.format(name="The jacobian of the cumulative hazard function")
    class_obj.jac_sf.__doc__ = JAC_BASE_DOCSTRING.format(name="The jacobian of the survival function")
    class_obj.jac_pdf.__doc__ = JAC_BASE_DOCSTRING.format(name="The jacobian of the probability density function")
    class_obj.jac_cdf.__doc__ = JAC_BASE_DOCSTRING.format(name="The jacobian of the cumulative distribution function")

    class_obj.ppf.__doc__ = PROBABILITY_BASE_DOCSTRING.format(name="The percent point function")
    class_obj.ppf.__doc__ += f"""
    Notes
    -----
    The ``ppf`` is the inverse of :py:meth:`~{class_obj}.cdf`.
    """
    class_obj.isf.__doc__ = PROBABILITY_BASE_DOCSTRING.format(name="Inverse survival function")

    class_obj.rvs.__doc__ = """
    Random variable sampling.

    Parameters
    ----------
    size : int, (int,) or (int, int)
        Size of the generated sample. If size is ``n`` or ``(n,)``, n samples are generated. If size is ``(m,n)``, a 
        2d array of samples is generated. 
    return_event : bool, default is False
        If True, returns event indicators along with the sample time values.
    random_entry : bool, default is False
        If True, returns corresponding entry values of the sample time values.
    seed : optional int, default is None
        Random seed used to fix random sampling.

    Returns
    -------
    float, ndarray or tuple of float or ndarray
        The sample values. If either ``return_event`` or ``random_entry`` is True, returns a tuple containing
        the time values followed by event values, entry values or both.
    """

    class_obj.plot.__doc__ = """
    Provides access to plotting functionality for this distribution.
    """

    class_obj.ls_integrate.__doc__ = """
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

    class_obj.moment.__doc__ = """
    n-th order moment

    Parameters
    ----------
    n : order of the moment, at least 1.

    Returns
    -------
    np.float64
        n-th order moment.
    """
    class_obj.mean.__doc__ = MOMENT_BASE_DOCSTRING.format(name="The mean")
    class_obj.var.__doc__ = MOMENT_BASE_DOCSTRING.format(name="The variance")
    class_obj.median.__doc__ = MOMENT_BASE_DOCSTRING.format(name="The median")

    class_obj.ichf.__doc__ = """
    Inverse cumulative hazard function.
    
    Parameters
    ----------
    cumulative_hazard_rate : float or np.ndarray
        Cumulative hazard rate value(s) at which to compute the function.
        If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
    
    Returns
    -------
    np.float64 or np.ndarray
        Function values at each given cumulative hazard rate(s).
    """

    class_obj.fit.__doc__ = """
    Estimation of parameters.

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
    **kwargs
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
