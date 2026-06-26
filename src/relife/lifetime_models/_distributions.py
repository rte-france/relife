"""Lifetime distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    Any,
    Concatenate,
    Literal,
    Self,
    final,
)

import numpy as np
import numpydoc.docscrape as docscrape  # pyright: ignore[reportMissingTypeStubs]
from optype.numpy import Array, Array1D, ArrayND
from scipy.special import digamma, exp1, gamma, gammaincc, gammainccinv
from typing_extensions import override

from relife.base import FittingResults
from relife.likelihoods import LifetimeLikelihood
from relife.quadratures import (
    laguerre_quadrature,
)
from relife.typing import VT

from ._base import (
    ParametricLifetimeModel,
    document_args,
)


class LifetimeDistribution(ParametricLifetimeModel[()], ABC):
    """
    Base class for distribution model.
    """

    fitting_results: FittingResults | None

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def sf(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        return super().sf(time)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def isf(self, probability: VT) -> np.float64 | ArrayND[np.float64]:
        cumulative_hazard_rate = -np.log(
            np.clip(probability, 0, 1 - np.finfo(float).resolution)
        )
        return self.ichf(cumulative_hazard_rate)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def cdf(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        return super().cdf(time)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def pdf(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        return super().pdf(time)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def ppf(self, probability: VT) -> np.float64 | ArrayND[np.float64]:
        return super().ppf(probability)

    @override
    @document_args(
        base_cls=ParametricLifetimeModel,
        args_docstring=[],
        returns=[docscrape.Parameter("out", "np.float64", [""])],
    )
    def median(self) -> np.float64 | ArrayND[np.float64]:
        return self.ppf(0.5)  # no super here to return np.float64

    @abstractmethod
    def jac_hf(
        self,
        time: VT,
    ) -> ArrayND[np.float64]:
        """
        The jacobian of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray

        Returns
        -------
        out : np.ndarray
        """

    @abstractmethod
    def jac_chf(
        self,
        time: VT,
    ) -> ArrayND[np.float64]:
        """
        The jacobian of the cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray

        Returns
        -------
        out : np.ndarray
        """

    @abstractmethod
    def dhf(self, time: VT) -> ArrayND[np.float64]:
        """
        The derivate of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray

        Returns
        -------
        out : np.float64 or np.ndarray
        """

    def jac_sf(self, time: VT) -> ArrayND[np.float64]:
        """
        The derivate of the survival function.

        Parameters
        ----------
        time : float or np.ndarray

        Returns
        -------
        out : np.float64 or np.ndarray
        """
        jac_chf, sf = self.jac_chf(time), self.sf(time)
        return -jac_chf * sf

    def jac_cdf(self, time: VT) -> ArrayND[np.float64]:
        """
        The derivate of the cumulative distribution function.

        Parameters
        ----------
        time : float or np.ndarray

        Returns
        -------
        out : np.float64 or np.ndarray
        """
        return -self.jac_sf(time)

    def jac_pdf(self, time: VT) -> ArrayND[np.float64]:
        """
        The derivate of the probability density function.

        Parameters
        ----------
        time : float or np.ndarray

        Returns
        -------
        out : np.float64 or np.ndarray
        """
        jac_hf, hf = self.jac_hf(time), self.hf(time)
        jac_sf, sf = self.jac_sf(time), self.sf(time)
        return jac_hf * sf + jac_sf * hf

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().rvs(
            size,
            seed=seed,
        )

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def ls_integrate(
        self,
        func: Callable[
            Concatenate[VT, ...],
            np.float64 | ArrayND[np.float64],
        ],
        a: VT,
        b: VT,
        *,
        deg: int = 10,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().ls_integrate(func, a, b, deg=deg)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def moment(self, n: int) -> np.float64:
        return np.float64(super().moment(n))

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def mean(self) -> np.float64:
        return np.float64(super().mean())

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def var(self) -> np.float64:
        return np.float64(super().var())

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def mrl(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        return super().mrl(time)

    def fit(
        self,
        time: Array1D[np.float64] | Array[tuple[int, Literal[2]], np.float64],
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> Self:

        optimizer = LifetimeLikelihood.from_data(
            self, time, event=event, entry=entry, **kwargs
        )
        self.fitting_results = optimizer.optimize()
        self.set_params(self.fitting_results.optimal_params)

        return self


@final
class Exponential(LifetimeDistribution):
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
    rate : float, default is None
        Rate parameter.

    Attributes
    ----------
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.).
        If the model is not fitted, the value is None.
    """

    def __init__(self, rate: float = np.nan):
        super().__init__([rate])

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def hf(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        return self.get_params()[0] * np.ones_like(time)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def chf(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        return self.get_params()[0] * time

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def ichf(self, cumulative_hazard_rate: VT) -> np.float64 | ArrayND[np.float64]:
        return cumulative_hazard_rate / self.get_params()[0]

    @override
    def jac_hf(self, time: VT) -> ArrayND[np.float64]:
        if isinstance(time, np.ndarray):
            jac = np.expand_dims(np.ones_like(time, dtype=np.float64), axis=0)
        else:
            jac = np.array([1], dtype=np.float64)
        return jac

    @override
    def jac_chf(self, time: VT) -> ArrayND[np.float64]:
        if isinstance(time, np.ndarray):
            jac = np.expand_dims(time, axis=0).astype(np.float64)
        else:
            jac = np.array([time], dtype=np.float64)
        return jac

    @override
    def dhf(self, time: VT) -> ArrayND[np.float64]:
        if isinstance(time, np.ndarray):
            return np.zeros_like(time, dtype=np.float64)
        return np.asarray(0, dtype=np.float64)

    @override
    def mean(self) -> np.float64:
        return 1 / self.get_params()[0]

    @override
    def var(self) -> np.float64:
        return 1 / self.get_params()[0] ** 2

    @override
    def mrl(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        return 1 / self.get_params()[0] * np.ones_like(time)

    @override
    def __repr__(self) -> str:
        return f"Exponential(rate={self.get_params()[0]!r})"


@final
class Weibull(LifetimeDistribution):
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
    shape : float, default is None
        Shape parameter.
    rate : float, default is None
        Rate parameter.

    Attributes
    ----------
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.).
        If the model is not fitted, the value is None.
    nb_params
    params
    params_names
    plot
    shape
    rate
    """

    def __init__(self, shape: float = np.nan, rate: float = np.nan):
        super().__init__([shape, rate])

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def hf(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        shape, rate = self.get_params()
        return shape * rate * (rate * np.asarray(time)) ** (shape - 1)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def chf(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        shape, rate = self.get_params()
        return (rate * np.asarray(time)) ** shape

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def ichf(self, cumulative_hazard_rate: VT) -> np.float64 | ArrayND[np.float64]:
        shape, rate = self.get_params()
        return np.asarray(cumulative_hazard_rate) ** (1 / shape) / rate

    @override
    def jac_hf(self, time: VT) -> ArrayND[np.float64]:
        shape, rate = self.get_params()
        return np.stack(
            (
                rate * (rate * time) ** (shape - 1) * (1 + shape * np.log(rate * time)),
                shape**2 * (rate * time) ** (shape - 1),
            ),
        )

    @override
    def jac_chf(self, time: VT) -> ArrayND[np.float64]:
        shape, rate = self.get_params()
        return np.stack(
            (
                np.log(rate * time) * (rate * time) ** shape,
                shape * time * (rate * time) ** (shape - 1),
            ),
        )

    @override
    def dhf(self, time: VT) -> ArrayND[np.float64]:
        shape, rate = self.get_params()
        return np.asarray(
            shape * (shape - 1) * rate**2 * (rate * time) ** (shape - 2),
        )

    @override
    def mean(self) -> np.float64:
        shape, rate = self.get_params()
        return gamma(1 + 1 / shape) / rate

    @override
    def var(self) -> np.float64:
        shape, rate = self.get_params()
        return gamma(1 + 2 / shape) / rate**2 - self.mean() ** 2

    @override
    def mrl(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        shape, rate = self.get_params()
        return (
            gamma(1 / shape)
            / (rate * shape * self.sf(time))
            * gammaincc(
                1 / shape,
                (rate * time) ** shape,
            )
        )

    @override
    def __repr__(self) -> str:
        params = self.get_params()
        return f"Weibull(shape={params[0]!r}, rate={params[1]!r})"


@final
class Gompertz(LifetimeDistribution):
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
    shape : float, default is None
        Shape parameter.
    rate : float, default is None
        Rate parameter.

    Attributes
    ----------
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.).
        If the model is not fitted, the value is None.
    """

    def __init__(self, shape: float = np.nan, rate: float = np.nan):
        super().__init__([shape, rate])

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def hf(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        shape, rate = self.get_params()
        return shape * rate * np.exp(rate * time)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def chf(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        shape, rate = self.get_params()
        return shape * np.expm1(rate * time)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def ichf(self, cumulative_hazard_rate: VT) -> np.float64 | ArrayND[np.float64]:
        shape, rate = self.get_params()
        return 1 / rate * np.log1p(cumulative_hazard_rate / shape)

    @override
    def jac_hf(self, time: VT) -> ArrayND[np.float64]:
        shape, rate = self.get_params()
        return np.stack(
            (
                rate * np.exp(rate * time),
                shape * np.exp(rate * time) * (1 + rate * time),
            ),
        )

    @override
    def jac_chf(self, time: VT) -> ArrayND[np.float64]:
        shape, rate = self.get_params()
        return np.stack(
            (
                np.expm1(rate * time),
                shape * time * np.exp(rate * time),
            ),
        )

    @override
    def dhf(self, time: VT) -> ArrayND[np.float64]:
        shape, rate = self.get_params()
        return shape * rate**2 * np.exp(rate * time)

    @override
    def mean(self) -> np.float64:
        shape, rate = self.get_params()
        return np.exp(shape) * exp1(shape) / rate

    @override
    def var(self) -> np.float64:
        return super().var()

    @override
    def mrl(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        shape, rate = self.get_params()
        z = shape * np.exp(rate * time)
        return np.exp(z) * exp1(z) / rate

    @override
    def __repr__(self) -> str:
        params = self.get_params()
        return f"Gompertz(shape={params[0]!r}, rate={params[1]!r})"


@final
class Gamma(LifetimeDistribution):
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
    shape : float, default is None
        Shape parameter.
    rate : float, default is None
        Rate parameter.

    Attributes
    ----------
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.).
        If the model is not fitted, the value is None.
    """

    def __init__(self, shape: float = np.nan, rate: float = np.nan):
        super().__init__([shape, rate])

    def _uppergamma(self, x: VT) -> np.float64 | ArrayND[np.float64]:
        shape, _ = self.get_params()
        x = np.asarray(x, dtype=np.float64)
        return gammaincc(shape, x) * gamma(shape)

    def _jac_uppergamma_shape(self, x: VT) -> np.float64 | ArrayND[np.float64]:
        shape, _ = self.get_params()

        def func(
            s: VT,
        ) -> np.float64 | ArrayND[np.float64]:
            return np.log(s) * s ** (shape - 1)

        return laguerre_quadrature(func, x, deg=100)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def hf(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        shape, rate = self.get_params()
        x = np.asarray(rate * time)
        return rate * x ** (shape - 1) * np.exp(-x) / self._uppergamma(x)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def chf(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        shape, rate = self.get_params()
        x = np.asarray(rate * time)
        return np.log(gamma(shape)) - np.log(self._uppergamma(x))

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def ichf(self, cumulative_hazard_rate: VT) -> np.float64 | ArrayND[np.float64]:
        shape, rate = self.get_params()
        return (
            1
            / rate
            * np.asarray(
                gammainccinv(shape, np.exp(-cumulative_hazard_rate)),
            )
        )

    @override
    def jac_hf(self, time: VT) -> ArrayND[np.float64]:
        shape, rate = self.get_params()
        x = rate * time
        y = x ** (shape - 1) * np.exp(-x) / self._uppergamma(x) ** 2
        jac = (
            y
            * (
                (rate * np.log(x) * self._uppergamma(x))
                - rate * self._jac_uppergamma_shape(x)
            ),
            y * ((shape - x) * self._uppergamma(x) + x**shape * np.exp(-x)),
        )
        return np.stack(jac)

    @override
    def jac_chf(self, time: VT) -> ArrayND[np.float64]:
        shape, rate = self.get_params()
        x = rate * time
        jac = (
            digamma(shape) - self._jac_uppergamma_shape(x) / self._uppergamma(x),
            (x ** (shape - 1) * time * np.exp(-x) / self._uppergamma(x)),
        )
        return np.stack(jac)

    @override
    def dhf(self, time: VT) -> ArrayND[np.float64]:
        shape, rate = self.get_params()
        return np.asarray(
            self.hf(time) * ((shape - 1) / time - rate + self.hf(time)),
        )

    @override
    def mean(self) -> np.float64:
        shape, rate = self.get_params()
        return shape / rate

    @override
    def var(self) -> np.float64:
        shape, rate = self.get_params()
        return shape / (rate**2)

    @override
    def __repr__(self) -> str:
        params = self.get_params()
        return f"Gamma(shape={params[0]!r}, rate={params[1]!r})"


@final
class LogLogistic(LifetimeDistribution):
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
    shape : float, default is None
        Shape parameter.
    rate : float, default is None
        Rate parameter.

    Attributes
    ----------
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.).
        If the model is not fitted, the value is None.
    """

    def __init__(self, shape: float = np.nan, rate: float = np.nan):
        super().__init__([shape, rate])

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def hf(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        shape, rate = self.get_params()
        x = rate * np.asarray(time)
        return shape * rate * x ** (shape - 1) / (1 + x**shape)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def chf(self, time: VT) -> np.float64 | ArrayND[np.float64]:
        shape, rate = self.get_params()
        x = rate * time
        return np.log(1 + x**shape)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def ichf(self, cumulative_hazard_rate: VT) -> np.float64 | ArrayND[np.float64]:
        shape, rate = self.get_params()
        return ((np.exp(cumulative_hazard_rate) - 1) ** (1 / shape)) / rate

    @override
    def jac_hf(self, time: VT) -> ArrayND[np.float64]:
        shape, rate = self.get_params()
        x = rate * time
        jac = (
            (rate * x ** (shape - 1) / (1 + x**shape) ** 2)
            * (1 + x**shape + shape * np.log(rate * time)),
            (rate * x ** (shape - 1) / (1 + x**shape) ** 2) * (shape**2 / rate),
        )
        return np.stack(jac)

    @override
    def jac_chf(self, time: VT) -> ArrayND[np.float64]:
        shape, rate = self.get_params()
        x = rate * time
        jac = (
            (x**shape / (1 + x**shape)) * np.log(rate * time),
            (x**shape / (1 + x**shape)) * (shape / rate),
        )
        return np.stack(jac)

    @override
    def dhf(self, time: VT) -> ArrayND[np.float64]:
        shape, rate = self.get_params()
        x = rate * np.asarray(time)
        return (
            shape
            * rate**2
            * x ** (shape - 2)
            * (shape - 1 - x**shape)
            / (1 + x**shape) ** 2
        )

    @override
    def mean(self) -> np.float64:
        shape, rate = self.get_params()
        b = np.pi / shape
        if shape <= 1:
            raise ValueError(f"Expectancy only defined for shape > 1: shape = {shape}")
        return b / (rate * np.sin(b))

    @override
    def var(self) -> np.float64:
        shape, rate = self.get_params()
        b = np.pi / shape
        if shape <= 2:
            raise ValueError(f"Variance only defined for shape > 2: shape = {shape}")
        return (1 / rate**2) * (2 * b / np.sin(2 * b) - b**2 / (np.sin(b) ** 2))

    @override
    def __repr__(self) -> str:
        params = self.get_params()
        return f"LogLogistic(shape={params[0]!r}, rate={params[1]!r})"
