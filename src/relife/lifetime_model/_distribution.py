"""Lifetime distributions."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import (
    Any,
    Self,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    final,
)

import numpy as np
import numpydoc.docscrape as docscrape  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from optype.numpy import Array1D, Array2D, ArrayND
from scipy.optimize import Bounds, newton
from scipy.special import digamma, exp1, gamma, gammaincc, gammainccinv
from typing_extensions import override

from relife.base import OptimizerConfig
from relife.quadrature import laguerre_quadrature, legendre_quadrature
from relife.utils import to_column_2d_if_1d

from ._base import (
    FittableParametricLifetimeModel,
    LifetimeData,
    LifetimeLikelihood,
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


ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint
M = TypeVar(
    "M",
    bound=FittableParametricLifetimeModel[()],
)


class LifetimeDistribution(FittableParametricLifetimeModel[()], ABC):
    """
    Base class for distribution model.
    """

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def sf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return super().sf(time)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def isf(
        self, probability: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        cumulative_hazard_rate = -np.log(
            np.clip(probability, 0, 1 - np.finfo(float).resolution)
        )
        return self.ichf(cumulative_hazard_rate)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def cdf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return super().cdf(time)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def pdf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return super().pdf(time)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def ppf(
        self, probability: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return super().ppf(probability)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def moment(self, n: int) -> np.float64 | ArrayND[np.float64]:
        return super().moment(n)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel,
        args_docstring=[],
        returns=[docscrape.Parameter("out", "np.float64", [""])],
    )
    def median(self) -> np.float64 | ArrayND[np.float64]:
        return self.ppf(0.5)  # no super here to return np.float64

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def jac_sf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        jac_chf, sf = self.jac_chf(time), self.sf(time)
        return -jac_chf * sf

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def jac_cdf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        return super().jac_cdf(time)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def jac_pdf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        jac_hf, hf = self.jac_hf(time), self.hf(time)
        jac_sf, sf = self.jac_sf(time), self.sf(time)
        return jac_hf * sf + jac_sf * hf

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def rvs(
        self,
        size: int | tuple[int, int],
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
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def ls_integrate(
        self,
        func: Callable[
            [ST | NumpyST | ArrayND[NumpyST]],
            np.float64 | ArrayND[np.float64],
        ],
        a: ST | NumpyST | ArrayND[NumpyST],
        b: ST | NumpyST | ArrayND[NumpyST],
        *,
        deg: int = 10,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().ls_integrate(func, a, b, deg=deg)

    @override
    def init_likelihood(
        self,
        time: Array1D[np.float64],
        args: Array1D[Any]
        | Array2D[Any]
        | tuple[Array1D[Any] | Array2D[Any], ...]
        | None = None,
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> LifetimeLikelihood[Self]:
        assert args is None
        lifetime_data = LifetimeData(time, event=event, entry=entry)
        x0 = kwargs.get("x0", init_distrib_params_from_lifetimes(self, lifetime_data))
        config = OptimizerConfig(x0)
        config.scipy_minimize_options["bounds"] = kwargs.get(
            "bounds", get_distrib_params_bounds(self)
        )
        config.scipy_minimize_options["method"] = kwargs.get("method", "L-BFGS-B")
        config.covariance_method = kwargs.get(
            "covariance_method", "2point" if isinstance(self, Gamma) else "cs"
        )
        optimizer = LifetimeLikelihood(self, lifetime_data, config)
        return optimizer


def init_distrib_params_from_lifetimes(
    model: LifetimeDistribution, data: LifetimeData
) -> Array1D[np.float64]:
    # flatten censored_time in case it is 2D
    all_time_values = np.concatenate(
        (data.complete_time.flatten(), data.censored_time.flatten())
    )
    if isinstance(model, Gompertz):
        param0 = np.empty(model.nb_params, dtype=np.float64)
        rate = np.pi / (np.sqrt(6) * np.std(all_time_values))
        shape = np.exp(-rate * np.mean(all_time_values))
        param0[0] = shape
        param0[1] = rate
        return param0

    param0 = np.ones(model.nb_params, dtype=np.float64)
    param0[-1] = 1 / np.median(all_time_values)
    return param0


def get_distrib_params_bounds(model: LifetimeDistribution) -> Bounds:
    return Bounds(
        np.full(model.nb_params, np.finfo(float).resolution),
        np.full(model.nb_params, np.inf),
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

    def __init__(self, rate: ST | None = None):
        super().__init__(rate=rate)

    @property
    def rate(self) -> np.float64:
        """
        Get the rate value.

        Returns
        -------
        out : float
        """
        return self.get_params()[0]

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def hf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return self.rate * np.ones_like(time)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def chf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return self.rate * time

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel,
        args_docstring=[],
        returns=[docscrape.Parameter("out", "np.float64", [""])],
    )
    def mean(self) -> np.float64 | ArrayND[np.float64]:
        return 1 / np.float64(self.rate)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel,
        args_docstring=[],
        returns=[docscrape.Parameter("out", "np.float64", [""])],
    )
    def var(self) -> np.float64 | ArrayND[np.float64]:
        return 1 / np.float64(self.rate) ** 2

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def mrl(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return 1 / self.rate * np.ones_like(time)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def ichf(
        self, cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return cumulative_hazard_rate / np.asarray(self.rate)

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def jac_hf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        if isinstance(time, np.ndarray):
            jac = np.expand_dims(np.ones_like(time, dtype=np.float64), axis=0).copy()
        else:
            jac = np.array([1], dtype=np.float64)
        return jac

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def jac_chf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        if isinstance(time, np.ndarray):
            jac = np.expand_dims(time, axis=0).copy().astype(np.float64)
        else:
            jac = np.array([time], dtype=np.float64)
        return jac

    @override
    @document_args(base_cls=FittableParametricLifetimeModel, args_docstring=[])
    def dhf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        if isinstance(time, np.ndarray):
            return np.zeros_like(time, dtype=np.float64)
        return np.asarray(0, dtype=np.float64)


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

    def __init__(self, shape: ST | None = None, rate: float | None = None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self) -> np.float64:
        """
        Get the shape value.

        Returns
        -------
        out : float
        """
        return self.get_params()[0]

    @property
    def rate(self) -> np.float64:
        """
        Returns the rate value.

        Returns
        -------
        out : float
        """
        return self.get_params()[1]

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def hf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return (
            self.shape * self.rate * (self.rate * np.asarray(time)) ** (self.shape - 1)
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def chf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return (self.rate * np.asarray(time)) ** self.shape

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel,
        args_docstring=[],
        returns=[docscrape.Parameter("out", "np.float64", [""])],
    )
    def mean(self) -> np.float64 | ArrayND[np.float64]:
        return gamma(1 + 1 / self.shape) / self.rate

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel,
        args_docstring=[],
        returns=[docscrape.Parameter("out", "np.float64", [""])],
    )
    def var(self) -> np.float64 | ArrayND[np.float64]:
        return gamma(1 + 2 / self.shape) / self.rate**2 - self.mean() ** 2

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def mrl(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return (
            gamma(1 / self.shape)
            / (self.rate * self.shape * self.sf(time))
            * gammaincc(
                1 / self.shape,
                (self.rate * time) ** self.shape,
            )
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def ichf(
        self, cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return np.asarray(cumulative_hazard_rate) ** (1 / self.shape) / self.rate

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def jac_hf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        return np.stack(
            (
                self.rate
                * (self.rate * time) ** (self.shape - 1)
                * (1 + self.shape * np.log(self.rate * time)),
                self.shape**2 * (self.rate * time) ** (self.shape - 1),
            ),
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def jac_chf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        return np.stack(
            (
                np.log(self.rate * time) * (self.rate * time) ** self.shape,
                self.shape * time * (self.rate * time) ** (self.shape - 1),
            ),
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def dhf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        return np.asarray(
            self.shape
            * (self.shape - 1)
            * self.rate**2
            * (self.rate * time) ** (self.shape - 2),
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

    def __init__(self, shape: ST | None = None, rate: float | None = None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self) -> np.float64:
        """
        Get the shape value.

        Returns
        -------
        out : float
        """
        return self.get_params()[0]

    @property
    def rate(self) -> np.float64:
        """
        Get the rate value.

        Returns
        -------
        out : float
        """
        return self.get_params()[1]

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def hf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return self.shape * self.rate * np.exp(self.rate * time)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def chf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return self.shape * np.expm1(self.rate * time)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel,
        args_docstring=[],
        returns=[docscrape.Parameter("out", "np.float64", [""])],
    )
    def mean(self) -> np.float64 | ArrayND[np.float64]:
        return np.exp(self.shape) * exp1(self.shape) / self.rate

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel,
        args_docstring=[],
        returns=[docscrape.Parameter("out", "np.float64", [""])],
    )
    def var(self) -> np.float64 | ArrayND[np.float64]:
        return super().var()

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def mrl(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        z = self.shape * np.exp(self.rate * time)
        return np.exp(z) * exp1(z) / self.rate

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def ichf(
        self, cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return 1 / self.rate * np.log1p(cumulative_hazard_rate / self.shape)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def jac_hf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        return np.stack(
            (
                self.rate * np.exp(self.rate * time),
                self.shape * np.exp(self.rate * time) * (1 + self.rate * time),
            ),
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def jac_chf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        return np.stack(
            (
                np.expm1(self.rate * time),
                self.shape * time * np.exp(self.rate * time),
            ),
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def dhf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        return self.shape * self.rate**2 * np.exp(self.rate * time)


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

    def __init__(self, shape: ST | None = None, rate: float | None = None):
        super().__init__(shape=shape, rate=rate)

    def _uppergamma(
        self, x: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        x = np.asarray(x, dtype=np.float64)
        return gammaincc(self.shape, x) * gamma(self.shape)

    def _jac_uppergamma_shape(
        self, x: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:

        def func(
            s: ST | NumpyST | ArrayND[NumpyST],
        ) -> np.float64 | ArrayND[np.float64]:
            return np.log(s) * s ** (self.shape - 1)

        return laguerre_quadrature(func, x, deg=100)

    @property
    def shape(self) -> np.float64:
        """
        Get the shape value.

        Returns
        -------
        out : float
        """
        return self.get_params()[0]

    @property
    def rate(self) -> np.float64:
        """
        Get the rate value.

        Returns
        -------
        out : float
        """
        return self.get_params()[1]

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def hf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        x = np.asarray(self.rate * time)
        return self.rate * x ** (self.shape - 1) * np.exp(-x) / self._uppergamma(x)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def chf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        x = np.asarray(self.rate * time)
        return np.log(gamma(self.shape)) - np.log(self._uppergamma(x))

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel,
        args_docstring=[],
        returns=[docscrape.Parameter("out", "np.float64", [""])],
    )
    def mean(self) -> np.float64 | ArrayND[np.float64]:
        return self.shape / self.rate

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel,
        args_docstring=[],
        returns=[docscrape.Parameter("out", "np.float64", [""])],
    )
    def var(self) -> np.float64 | ArrayND[np.float64]:
        return self.shape / (self.rate**2)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def ichf(
        self, cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return (
            1
            / self.rate
            * np.asarray(
                gammainccinv(self.shape, np.exp(-cumulative_hazard_rate)),
            )
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def jac_hf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
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
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def jac_chf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        x = self.rate * time
        jac = (
            digamma(self.shape) - self._jac_uppergamma_shape(x) / self._uppergamma(x),
            (x ** (self.shape - 1) * time * np.exp(-x) / self._uppergamma(x)),
        )
        return np.stack(jac)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def dhf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        return np.asarray(
            self.hf(time) * ((self.shape - 1) / time - self.rate + self.hf(time)),
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def mrl(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
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

    def __init__(self, shape: ST | None = None, rate: float | None = None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self) -> np.float64:
        """
        Get the shape value.

        Returns
        -------
        out : float
        """
        return self.get_params()[0]

    @property
    def rate(self) -> np.float64:
        """
        Get the rate value.

        Returns
        -------
        out : float
        """
        return self.get_params()[1]

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def hf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        x = self.rate * np.asarray(time)
        return self.shape * self.rate * x ** (self.shape - 1) / (1 + x**self.shape)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def chf(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        x = self.rate * time
        return np.log(1 + x**self.shape)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel,
        args_docstring=[],
        returns=[docscrape.Parameter("out", "np.float64", [""])],
    )
    def mean(self) -> np.float64 | ArrayND[np.float64]:
        b = np.pi / self.shape
        if self.shape <= 1:
            raise ValueError(
                f"Expectancy only defined for shape > 1: shape = {self.shape}"
            )
        return b / (self.rate * np.sin(b))

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel,
        args_docstring=[],
        returns=[docscrape.Parameter("out", "np.float64", [""])],
    )
    def var(self) -> np.float64 | ArrayND[np.float64]:
        b = np.pi / self.shape
        if self.shape <= 2:
            raise ValueError(
                f"Variance only defined for shape > 2: shape = {self.shape}"
            )
        return (1 / self.rate**2) * (2 * b / np.sin(2 * b) - b**2 / (np.sin(b) ** 2))

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def ichf(
        self, cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        return ((np.exp(cumulative_hazard_rate) - 1) ** (1 / self.shape)) / self.rate

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def jac_hf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        x = self.rate * time
        jac = (
            (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2)
            * (1 + x**self.shape + self.shape * np.log(self.rate * time)),
            (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2)
            * (self.shape**2 / self.rate),
        )
        return np.stack(jac)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def jac_chf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        x = self.rate * time
        jac = (
            (x**self.shape / (1 + x**self.shape)) * np.log(self.rate * time),
            (x**self.shape / (1 + x**self.shape)) * (self.shape / self.rate),
        )
        return np.stack(jac)

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def dhf(self, time: ST | NumpyST | ArrayND[NumpyST]) -> ArrayND[np.float64]:
        x = self.rate * np.asarray(time)
        return (
            self.shape
            * self.rate**2
            * x ** (self.shape - 2)
            * (self.shape - 1 - x**self.shape)
            / (1 + x**self.shape) ** 2
        )

    @override
    @document_args(base_cls=LifetimeDistribution, args_docstring=[])
    def mrl(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
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
    def cdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
        return legendre_quadrature(
            lambda x: np.asarray(self.baseline.sf(x, *args), dtype=float), 0, time
        ) / self.baseline.mean(*args)

    @override
    def sf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
        return 1 - self.cdf(time, *args)

    @override
    def pdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
        return self.baseline.sf(time, *args) / self.baseline.mean(*args)

    @override
    def hf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
        return 1 / self.baseline.mrl(time, *args)

    @override
    def chf(
        self, time: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
        return -np.log(self.sf(time, *args))

    @override
    def isf(
        self, probability: ST | NumpyST | ArrayND[NumpyST], *args: *Ts
    ) -> np.float64 | ArrayND[np.float64]:
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
        cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST],
        *args: *Ts,
    ) -> np.float64 | ArrayND[np.float64]:
        return self.isf(np.exp(-cumulative_hazard_rate), *args)


AnyUnsignedInt: TypeAlias = int | np.uint | NDArray[np.uint]


@final
class MinimumDistribution(FittableParametricLifetimeModel[AnyUnsignedInt]):
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

    baseline: LifetimeDistribution

    def __init__(self, baseline: LifetimeDistribution):
        super().__init__()
        self.baseline = baseline

    @override
    def sf(
        self, time: ST | NumpyST | ArrayND[NumpyST], n: AnyUnsignedInt
    ) -> np.float64 | ArrayND[np.float64]:
        return super().sf(time, n)

    @override
    def pdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], n: AnyUnsignedInt
    ) -> np.float64 | ArrayND[np.float64]:
        return super().pdf(time, n)

    @override
    def hf(
        self, time: ST | NumpyST | ArrayND[NumpyST], n: AnyUnsignedInt
    ) -> np.float64 | ArrayND[np.float64]:
        return n * self.baseline.hf(time)

    @override
    def chf(
        self, time: ST | NumpyST | ArrayND[NumpyST], n: AnyUnsignedInt
    ) -> np.float64 | ArrayND[np.float64]:
        return n * self.baseline.chf(time)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: ST | NumpyST | ArrayND[NumpyST],
        n: AnyUnsignedInt,
    ) -> np.float64 | ArrayND[np.float64]:
        return self.baseline.ichf(cumulative_hazard_rate / n)

    @override
    def dhf(
        self, time: ST | NumpyST | ArrayND[NumpyST], n: AnyUnsignedInt
    ) -> ArrayND[np.float64]:
        return n * self.baseline.dhf(time)

    @override
    def jac_chf(
        self, time: ST | NumpyST | ArrayND[NumpyST], n: AnyUnsignedInt
    ) -> ArrayND[np.float64]:
        return n * self.baseline.jac_chf(time)

    @override
    def jac_hf(
        self, time: ST | NumpyST | ArrayND[NumpyST], n: AnyUnsignedInt
    ) -> ArrayND[np.float64]:
        return n * self.baseline.jac_chf(time)

    @override
    def jac_sf(
        self, time: ST | NumpyST | ArrayND[NumpyST], n: AnyUnsignedInt
    ) -> ArrayND[np.float64]:
        jac_chf, sf = (
            self.jac_chf(time, n),
            self.sf(time, n),
        )
        return -jac_chf * sf

    @override
    def jac_cdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], n: AnyUnsignedInt
    ) -> ArrayND[np.float64]:
        return super().jac_cdf(time, n)

    @override
    def jac_pdf(
        self, time: ST | NumpyST | ArrayND[NumpyST], n: AnyUnsignedInt
    ) -> ArrayND[np.float64]:
        jac_hf, hf = self.jac_hf(time, n), self.hf(time, n)
        jac_sf, sf = self.jac_sf(time, n), self.sf(time, n)
        return jac_hf * sf + jac_sf * hf

    @override
    def ls_integrate(
        self,
        func: Callable[
            [ST | NumpyST | ArrayND[NumpyST]],
            np.float64 | ArrayND[np.float64],
        ],
        a: ST | NumpyST | ArrayND[NumpyST],
        b: ST | NumpyST | ArrayND[NumpyST],
        n: AnyUnsignedInt,
        *,
        deg: int = 10,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().ls_integrate(func, a, b, n, deg=deg)

    @override
    def init_likelihood(
        self,
        time: Array1D[np.float64],
        args: Array1D[Any]
        | Array2D[Any]
        | tuple[Array1D[Any] | Array2D[Any], ...]
        | None = None,
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> LifetimeLikelihood[Self]:
        if not isinstance(args, np.ndarray):
            raise ValueError("args is expected to be covar only.")
        args = to_column_2d_if_1d(args)
        lifetime_data = LifetimeData(time, args, event, entry)
        x0 = kwargs.get(
            "x0", init_distrib_params_from_lifetimes(self.baseline, lifetime_data)
        )
        config = OptimizerConfig(x0)
        config.scipy_minimize_options["bounds"] = kwargs.get(
            "bounds", get_distrib_params_bounds(self.baseline)
        )
        config.scipy_minimize_options["method"] = kwargs.get("method", "L-BFGS-B")
        config.covariance_method = kwargs.get(
            "covariance_method", "2point" if isinstance(self.baseline, Gamma) else "cs"
        )
        optimizer = LifetimeLikelihood(self, lifetime_data, config)
        return optimizer

    @override
    def fit(
        self,
        time: Array1D[np.float64],
        args: Array1D[Any]
        | Array2D[Any]
        | tuple[Array1D[Any] | Array2D[Any], ...]
        | None = None,
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> Self:
        if not isinstance(args, np.ndarray):
            raise ValueError("args is expected to contain n values only.")
        return super().fit(
            time,
            args=args,
            event=event,
            entry=entry,
            **kwargs,
        )
