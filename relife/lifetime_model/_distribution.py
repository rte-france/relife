"""Lifetime distributions."""

from __future__ import annotations

from abc import ABC
from typing import (
    Any,
    Callable,
    Literal,
    Self,
    TypeAlias,
    TypeVarTuple,
    final,
    overload,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, newton
from scipy.special import digamma, exp1, gamma, gammaincc, gammainccinv
from typing_extensions import override

from relife.typing import AnyFloat, NumpyBool, NumpyFloat, ScipyMinimizeOptions, Seed
from relife.utils.quadrature import laguerre_quadrature, legendre_quadrature

from ._base import (
    FittableParametricLifetimeModel,
    ParametricLifetimeModel,
    document_args,
)

__all__: list[str] = [
    "Gompertz",
    "Weibull",
    "Gamma",
    "LogLogistic",
    "EquilibriumDistribution",
    "Exponential",
    "MinimumDistribution",
]


class LifetimeDistribution(FittableParametricLifetimeModel[()], ABC):
    """
    Base class for distribution model.
    """

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring="")
    def sf(self, time: AnyFloat) -> NumpyFloat:
        return super().sf(time)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring="")
    def isf(self, probability: AnyFloat) -> NumpyFloat:
        cumulative_hazard_rate = -np.log(
            np.clip(probability, 0, 1 - np.finfo(float).resolution)
        )
        return self.ichf(cumulative_hazard_rate)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring="")
    def cdf(self, time: AnyFloat) -> NumpyFloat:
        return super().cdf(time)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring="")
    def pdf(self, time: AnyFloat) -> NumpyFloat:
        return super().pdf(time)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring="")
    def ppf(self, probability: AnyFloat) -> NumpyFloat:
        return super().ppf(probability)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring="")
    def moment(self, n: int) -> NumpyFloat:
        return super().moment(n)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring="")
    def median(self) -> NumpyFloat:
        return self.ppf(0.5)  # no super here to return np.float64

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring="")
    def jac_sf(self, time: AnyFloat) -> NumpyFloat:
        jac_chf, sf = self.jac_chf(time), self.sf(time)
        return -jac_chf * sf

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring="")
    def jac_cdf(self, time: AnyFloat) -> NumpyFloat:
        return super().jac_cdf(time)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring="")
    def jac_pdf(self, time: AnyFloat) -> NumpyFloat:
        jac_hf, hf = self.jac_hf(time), self.hf(time)
        jac_sf, sf = self.jac_sf(time), self.sf(time)
        return jac_hf * sf + jac_sf * hf

    @overload
    def rvs(
        self,
        size: int | tuple[int, int],
        *,
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> NumpyFloat: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int, int],
        *,
        return_event: Literal[True],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool]: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int, int],
        *,
        return_event: Literal[False],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int | tuple[int, int],
        *,
        return_event: Literal[True],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool, NumpyFloat]: ...
    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring="")
    def rvs(
        self,
        size: int | tuple[int, int],
        *,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Seed | None = None,
    ) -> (
        NumpyFloat
        | tuple[NumpyFloat, NumpyBool]
        | tuple[NumpyFloat, NumpyFloat]
        | tuple[NumpyFloat, NumpyBool, NumpyFloat]
    ):
        return super().rvs(
            size,
            return_event=return_event,
            return_entry=return_entry,
            seed=seed,
        )

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring="")
    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: AnyFloat,
        b: AnyFloat,
        *,
        deg: int = 10,
    ) -> NumpyFloat:
        return super().ls_integrate(func, a, b, deg=deg)

    @override
    def get_initial_params(
        self,
        time: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
    ) -> NDArray[np.float64]:
        param0 = np.ones(self.nb_params, dtype=np.float64)
        param0[-1] = 1 / np.median(time)
        return param0

    @property
    @override
    def params_bounds(self) -> Bounds:
        return Bounds(
            np.full(self.nb_params, np.finfo(float).resolution),
            np.full(self.nb_params, np.inf),
        )

    @override
    def fit(
        self,
        time: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
        optimizer_options: ScipyMinimizeOptions | None = None,
    ) -> Self:
        if model_args is not None:
            raise ValueError(
                "LifetimeDistribution does not expect additional arguments in model_args"
            )
        return super().fit(
            time, event=event, entry=entry, optimizer_options=optimizer_options
        )

    @override
    def fit_from_interval_censored_lifetimes(
        self,
        time_inf: NDArray[np.float64],
        time_sup: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
        entry: NDArray[np.float64] | None = None,
        optimizer_options: ScipyMinimizeOptions | None = None,
    ) -> Self:
        if model_args is not None:
            raise ValueError(
                "LifetimeDistribution does not expect additional arguments in model_args"
            )
        return super().fit_from_interval_censored_lifetimes(
            time_inf, time_sup, entry=entry, optimizer_options=optimizer_options
        )


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
    nb_params
    params
    params_names
    plot
    rate
    """

    def __init__(self, rate: float | None = None):
        super().__init__(rate=rate)

    @property
    def rate(self):  # optional but better for clarity and type checking
        """
        Returns the rate value.

        Returns
        -------
        out: float
        """
        return self._params["rate"]

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def hf(self, time: AnyFloat) -> NumpyFloat:
        return self.rate * np.ones_like(time)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def chf(self, time: AnyFloat) -> NumpyFloat:
        return np.asarray(self.rate, dtype=np.float64) * time

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def mean(self) -> NumpyFloat:
        return 1 / np.asarray(self.rate)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def var(self) -> NumpyFloat:
        return 1 / np.asarray(self.rate) ** 2

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def mrl(self, time: AnyFloat) -> NumpyFloat:
        return 1 / self.rate * np.ones_like(time)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def ichf(self, cumulative_hazard_rate: AnyFloat) -> NumpyFloat:
        return cumulative_hazard_rate / np.asarray(self.rate)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def jac_hf(self, time: AnyFloat) -> NumpyFloat:
        if isinstance(time, np.ndarray):
            jac = np.expand_dims(np.ones_like(time, dtype=np.float64), axis=0).copy()
        else:
            jac = np.array([1], dtype=np.float64)
        return jac

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def jac_chf(self, time: AnyFloat) -> NumpyFloat:
        if isinstance(time, np.ndarray):
            jac = np.expand_dims(time, axis=0).copy().astype(np.float64)
        else:
            jac = np.array([time], dtype=np.float64)
        return jac

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def dhf(self, time: AnyFloat) -> NumpyFloat:
        if isinstance(time, np.ndarray):
            return np.zeros_like(time, dtype=np.float64)
        return np.asarray(0.0)


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

    def __init__(self, shape: float | None = None, rate: float | None = None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self) -> float:  # optional but better for clarity and type checking
        """
        Returns shape value.

        Returns
        -------
        out: float
        """
        return self._params["shape"]

    @property
    def rate(self) -> float:  # optional but better for clarity and type checking
        """
        Returns the rate value.

        Returns
        -------
        out: float
        """
        return self._params["rate"]

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def hf(self, time: AnyFloat) -> NumpyFloat:
        return (
            self.shape * self.rate * (self.rate * np.asarray(time)) ** (self.shape - 1)
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def chf(self, time: AnyFloat) -> NumpyFloat:
        return (self.rate * np.asarray(time)) ** self.shape

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def mean(self) -> NumpyFloat:
        return gamma(1 + 1 / self.shape) / self.rate

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def var(self) -> NumpyFloat:
        return gamma(1 + 2 / self.shape) / self.rate**2 - self.mean() ** 2

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def mrl(self, time: AnyFloat) -> NumpyFloat:
        return (
            gamma(1 / self.shape)
            / (self.rate * self.shape * self.sf(time))
            * gammaincc(
                1 / self.shape,
                (self.rate * time) ** self.shape,
            )
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def ichf(self, cumulative_hazard_rate: AnyFloat) -> NumpyFloat:
        return np.asarray(cumulative_hazard_rate) ** (1 / self.shape) / self.rate

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def jac_hf(self, time: AnyFloat) -> NumpyFloat:
        return np.stack(
            (
                self.rate
                * (self.rate * time) ** (self.shape - 1)
                * (1 + self.shape * np.log(self.rate * time)),
                self.shape**2 * (self.rate * time) ** (self.shape - 1),
            )
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def jac_chf(self, time: AnyFloat) -> NumpyFloat:
        return np.stack(
            (
                np.log(self.rate * time) * (self.rate * time) ** self.shape,
                self.shape * time * (self.rate * time) ** (self.shape - 1),
            )
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def dhf(self, time: AnyFloat) -> NumpyFloat:
        time = np.asarray(time)
        return (
            self.shape
            * (self.shape - 1)
            * self.rate**2
            * (self.rate * time) ** (self.shape - 2)
        )


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
    nb_params
    params
    params_names
    plot
    shape
    rate
    """

    def __init__(self, shape: float | None = None, rate: float | None = None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self) -> float:  # optional but better for clarity and type checking
        """
        Returns the shape value.

        Returns
        -------
        out: float
        """
        return self._params["shape"]

    @property
    def rate(self) -> float:  # optional but better for clarity and type checking
        """
        Returns the rate value.

        Returns
        -------
        out: float
        """
        return self._params["rate"]

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def hf(self, time: AnyFloat) -> NumpyFloat:
        return self.shape * self.rate * np.exp(self.rate * time)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def chf(self, time: AnyFloat) -> NumpyFloat:
        return self.shape * np.expm1(self.rate * time)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def mean(self) -> NumpyFloat:
        return np.exp(self.shape) * exp1(self.shape) / self.rate

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def var(self) -> NumpyFloat:
        return super().var()

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def mrl(self, time: AnyFloat) -> NumpyFloat:
        z = self.shape * np.exp(self.rate * time)
        return np.exp(z) * exp1(z) / self.rate

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def ichf(self, cumulative_hazard_rate: AnyFloat) -> NumpyFloat:
        return 1 / self.rate * np.log1p(cumulative_hazard_rate / self.shape)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def jac_hf(self, time: AnyFloat) -> NumpyFloat:
        return np.stack(
            (
                self.rate * np.exp(self.rate * time),
                self.shape * np.exp(self.rate * time) * (1 + self.rate * time),
            )
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def jac_chf(self, time: AnyFloat) -> NumpyFloat:
        return np.stack(
            (
                np.expm1(self.rate * time),
                self.shape * time * np.exp(self.rate * time),
            )
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def dhf(self, time: AnyFloat) -> NumpyFloat:
        return self.shape * self.rate**2 * np.exp(self.rate * time)

    @override
    def get_initial_params(
        self,
        time: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
    ) -> NDArray[np.float64]:
        param0 = np.empty(self.nb_params, dtype=np.float64)
        rate = np.pi / (np.sqrt(6) * np.std(time))
        shape = np.exp(-rate * np.mean(time))
        param0[0] = shape
        param0[1] = rate
        return param0


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
    nb_params
    params
    params_names
    plot
    shape
    rate
    """

    def __init__(self, shape: float | None = None, rate: float | None = None):
        super().__init__(shape=shape, rate=rate)

    def _uppergamma(self, x: AnyFloat) -> NumpyFloat:
        x = np.asarray(x, dtype=np.float64)
        return gammaincc(self.shape, x) * gamma(self.shape)

    def _jac_uppergamma_shape(self, x: AnyFloat) -> NumpyFloat:
        return laguerre_quadrature(
            lambda s: np.log(s) * s ** (self.shape - 1), x, deg=100
        )

    @property
    def shape(self) -> float:  # optional but better for clarity and type checking
        """
        Returns the shape value.

        Returns
        -------
        out: float
        """
        return self._params["shape"]

    @property
    def rate(self) -> float:  # optional but better for clarity and type checking
        """
        Returns the rate value.

        Returns
        -------
        out: float
        """
        return self._params["rate"]

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def hf(self, time: AnyFloat) -> NumpyFloat:
        x = np.asarray(self.rate * time, dtype=np.float64)
        return self.rate * x ** (self.shape - 1) * np.exp(-x) / self._uppergamma(x)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def chf(self, time: AnyFloat) -> NumpyFloat:
        x = np.asarray(self.rate * time, dtype=np.float64)
        return np.log(gamma(self.shape)) - np.log(self._uppergamma(x))

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def mean(self) -> NumpyFloat:
        return np.asarray(self.shape / self.rate)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def var(self) -> NumpyFloat:
        return np.asarray(self.shape / (self.rate**2))

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def ichf(self, cumulative_hazard_rate: AnyFloat) -> NumpyFloat:
        return (
            1
            / self.rate
            * np.asarray(
                gammainccinv(self.shape, np.exp(-cumulative_hazard_rate)),
                dtype=np.float64,
            )
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def jac_hf(self, time: AnyFloat) -> NumpyFloat:
        x = self.rate * time
        y = x ** (self.shape - 1) * np.exp(-x) / self._uppergamma(x) ** 2
        jac = (
            y
            * (
                (self.rate * np.log(x) * self._uppergamma(x))
                - self.rate * self._jac_uppergamma_shape(x)
            ),
            y * ((self.shape - x) * self._uppergamma(x) + x**self.shape * np.exp(-x)),
        )
        return np.stack(jac)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def jac_chf(self, time: AnyFloat) -> NumpyFloat:
        x = self.rate * time
        jac = (
            digamma(self.shape) - self._jac_uppergamma_shape(x) / self._uppergamma(x),
            (x ** (self.shape - 1) * time * np.exp(-x) / self._uppergamma(x)),
        )
        return np.stack(jac)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def dhf(self, time: AnyFloat) -> NumpyFloat:
        return self.hf(time) * ((self.shape - 1) / time - self.rate + self.hf(time))

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def mrl(self, time: AnyFloat) -> NumpyFloat:
        return super().mrl(time)


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
    nb_params
    params
    params_names
    plot
    shape
    rate
    """

    def __init__(self, shape: float | None = None, rate: float | None = None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self) -> float:  # optional but better for clarity and type checking
        """
        Returns the shape value.

        Returns
        -------
        out: float
        """
        return self._params["shape"]

    @property
    def rate(self) -> float:  # optional but better for clarity and type checking
        """Returns the rate value.

        Returns
        -------
        out: float
        """
        return self._params["rate"]

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def hf(self, time: AnyFloat) -> NumpyFloat:
        x = self.rate * np.asarray(time)
        return self.shape * self.rate * x ** (self.shape - 1) / (1 + x**self.shape)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def chf(self, time: AnyFloat) -> NumpyFloat:
        x = self.rate * time
        return np.log(1 + x**self.shape)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def mean(self) -> NumpyFloat:
        b = np.pi / self.shape
        if self.shape <= 1:
            raise ValueError(
                f"Expectancy only defined for shape > 1: shape = {self.shape}"
            )
        return b / (self.rate * np.sin(b))

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def var(self) -> NumpyFloat:
        b = np.pi / self.shape
        if self.shape <= 2:
            raise ValueError(
                f"Variance only defined for shape > 2: shape = {self.shape}"
            )
        return (1 / self.rate**2) * (2 * b / np.sin(2 * b) - b**2 / (np.sin(b) ** 2))

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def ichf(self, cumulative_hazard_rate: AnyFloat) -> NumpyFloat:
        return ((np.exp(cumulative_hazard_rate) - 1) ** (1 / self.shape)) / self.rate

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def jac_hf(self, time: AnyFloat) -> NumpyFloat:
        x = self.rate * time
        jac = (
            (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2)
            * (1 + x**self.shape + self.shape * np.log(self.rate * time)),
            (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2)
            * (self.shape**2 / self.rate),
        )
        return np.stack(jac)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def jac_chf(self, time: AnyFloat) -> NumpyFloat:
        x = self.rate * time
        jac = (
            (x**self.shape / (1 + x**self.shape)) * np.log(self.rate * time),
            (x**self.shape / (1 + x**self.shape)) * (self.shape / self.rate),
        )
        return np.stack(jac)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def dhf(self, time: AnyFloat) -> NumpyFloat:
        x = self.rate * np.asarray(time)
        return (
            self.shape
            * self.rate**2
            * x ** (self.shape - 2)
            * (self.shape - 1 - x**self.shape)
            / (1 + x**self.shape) ** 2
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring="")
    def mrl(self, time: AnyFloat) -> NumpyFloat:
        return super().mrl(time)


Ts = TypeVarTuple("Ts")


@final
class EquilibriumDistribution(ParametricLifetimeModel[*Ts]):
    r"""
    Equilibrium distribution.

    The equilibirum distribution is the distribution that makes the renewal process
    stationnary.

    Parameters
    ----------
    baseline : any parametric lifetime model
        Lifetime model.

    References
    ----------
    .. [1] Ross, S. M. (1996). Stochastic stochastic_process. New York: Wiley.
    """

    baseline: ParametricLifetimeModel[*Ts]

    def __init__(self, baseline: ParametricLifetimeModel[*Ts]):
        super().__init__()
        self.baseline = baseline

    @override
    def cdf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        return legendre_quadrature(
            lambda x: self.baseline.sf(x, *args), 0, time
        ) / self.baseline.mean(*args)

    @override
    def sf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        return 1 - self.cdf(time, *args)

    @override
    def pdf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        return self.baseline.sf(time, *args) / self.baseline.mean(*args)

    @override
    def hf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        return 1 / self.baseline.mrl(time, *args)

    @override
    def chf(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
        return -np.log(self.sf(time, *args))

    @override
    def isf(self, probability: AnyFloat, *args: *Ts) -> NumpyFloat:
        def func(x: NDArray[np.float64]) -> np.float64:
            return np.sum(self.sf(x, *args) - probability)

        return newton(
            func,
            x0=np.asarray(self.baseline.isf(probability, *args)),
            args=args,
        )

    @override
    def ichf(
        self,
        cumulative_hazard_rate: AnyFloat,
        *args: *Ts,
    ) -> NumpyFloat:
        return self.isf(np.exp(-cumulative_hazard_rate), *args)


AnyInt: TypeAlias = int | np.int64 | NDArray[np.int64]


@final
class MinimumDistribution(FittableParametricLifetimeModel[*tuple[AnyInt, *Ts]]):
    r"""
    Series structure of n identical and independent components.

    The hazard function of the system is given by:

    .. math::

        h(t) = n \cdot  h_0(t)

    where :math:`h_0` is the baseline hazard function of the components.

    Parameters
    ----------
    baseline : lifetime distribution or regression
        Lifetime model.

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

    baseline: FittableParametricLifetimeModel[*Ts]

    def __init__(self, baseline: FittableParametricLifetimeModel[*Ts]):
        super().__init__()
        self.baseline = baseline

    @override
    def sf(self, time: AnyFloat, n: AnyInt, *args: *Ts) -> NumpyFloat:
        return super().sf(time, *(n, *args))

    @override
    def pdf(self, time: AnyFloat, n: AnyInt, *args: *Ts) -> NumpyFloat:
        return super().pdf(time, *(n, *args))

    @override
    def hf(self, time: AnyFloat, n: AnyInt, *args: *Ts) -> NumpyFloat:
        return n * self.baseline.hf(time, *args)

    @override
    def chf(self, time: AnyFloat, n: AnyInt, *args: *Ts) -> NumpyFloat:
        return n * self.baseline.chf(time, *args)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: AnyFloat,
        n: AnyInt,
        *args: *Ts,
    ) -> NumpyFloat:
        return self.baseline.ichf(cumulative_hazard_rate / float(n), *args)

    @override
    def dhf(self, time: AnyFloat, n: AnyInt, *args: *Ts) -> NumpyFloat:
        return n * self.baseline.dhf(time, *args)

    @override
    def jac_chf(self, time: AnyFloat, n: AnyInt, *args: *Ts) -> NumpyFloat:
        return n * self.baseline.jac_chf(time, *args)

    @override
    def jac_hf(self, time: AnyFloat, n: AnyInt, *args: *Ts) -> NumpyFloat:
        return n * self.baseline.jac_chf(time, *args)

    @override
    def jac_sf(self, time: AnyFloat, n: AnyInt, *args: *Ts) -> NumpyFloat:
        jac_chf, sf = (
            self.jac_chf(time, n, *args),
            self.sf(time, n, *args),
        )
        return -jac_chf * sf

    @override
    def jac_cdf(self, time: AnyFloat, n: AnyInt, *args: *Ts) -> NumpyFloat:
        return super().jac_cdf(time, n, *args)

    @override
    def jac_pdf(self, time: AnyFloat, n: AnyInt, *args: *Ts) -> NumpyFloat:
        jac_hf, hf = self.jac_hf(time, n, *args), self.hf(time, n, *args)
        jac_sf, sf = self.jac_sf(time, n, *args), self.sf(time, n, *args)
        return jac_hf * sf + jac_sf * hf

    @override
    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: AnyFloat,
        b: AnyFloat,
        n: AnyInt,
        *args: *Ts,
        deg: int = 10,
    ) -> NumpyFloat:
        return super().ls_integrate(func, a, b, n, *args, deg=deg)

    @property
    @override
    def params_bounds(self) -> Bounds:
        return self.baseline.params_bounds

    @override
    def get_initial_params(
        self,
        time: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
    ) -> NDArray[np.float64]:
        return self.baseline.get_initial_params(time, model_args=model_args)

    @override
    def fit(
        self,
        time: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
        optimizer_options: ScipyMinimizeOptions | None = None,
    ) -> Self:
        if model_args is None:
            raise ValueError(
                "MinimumDistribution expects at least one additional argument in model_args"
            )
        return super().fit(
            time,
            model_args=model_args,
            event=event,
            entry=entry,
            optimizer_options=optimizer_options,
        )

    @override
    def fit_from_interval_censored_lifetimes(
        self,
        time_inf: NDArray[np.float64],
        time_sup: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
        entry: NDArray[np.float64] | None = None,
        optimizer_options: ScipyMinimizeOptions | None = None,
    ) -> Self:
        if model_args is None:
            raise ValueError(
                "MinimumDistribution expects at least one additional argument in model_args"
            )
        return super().fit_from_interval_censored_lifetimes(
            time_inf,
            time_sup,
            model_args=model_args,
            entry=entry,
            optimizer_options=optimizer_options,
        )
