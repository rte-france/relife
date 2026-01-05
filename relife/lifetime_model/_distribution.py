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
    def sf(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def isf(self, probability: AnyFloat) -> NumpyFloat:
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

    @override
    def cdf(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def pdf(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def ppf(self, probability: AnyFloat) -> NumpyFloat:
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

    @override
    def moment(self, n: int) -> NumpyFloat:
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

    @override
    def median(self) -> NumpyFloat:
        """
        The median.

        Returns
        -------
        np.float64
        """
        return self.ppf(0.5)  # no super here to return np.float64

    @overload
    def jac_sf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_sf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_sf(
        self,
        time: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_sf(self, time: AnyFloat, *, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
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

    @overload
    def jac_cdf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_cdf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_cdf(
        self,
        time: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    def jac_cdf(self, time: AnyFloat, *, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
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

    @overload
    def jac_pdf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_pdf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_pdf(
        self,
        time: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_pdf(self, time: AnyFloat, *, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
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

    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> NumpyFloat: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[False],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool]: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
        return_event: Literal[False],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
        return_event: Literal[True],
        return_entry: Literal[True],
        seed: Seed | None = None,
    ) -> tuple[NumpyFloat, NumpyBool, NumpyFloat]: ...
    @overload
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Seed | None = None,
    ) -> (
        NumpyFloat
        | tuple[NumpyFloat, NumpyBool]
        | tuple[NumpyFloat, NumpyFloat]
        | tuple[NumpyFloat, NumpyBool, NumpyFloat]
    ): ...
    @override
    def rvs(
        self,
        size: int,
        *,
        nb_assets: int | None = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Seed | None = None,
    ) -> (
        NumpyFloat
        | tuple[NumpyFloat, NumpyBool]
        | tuple[NumpyFloat, NumpyFloat]
        | tuple[NumpyFloat, NumpyBool, NumpyFloat]
    ):
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
        return super().rvs(
            size,
            nb_assets=nb_assets,
            return_event=return_event,
            return_entry=return_entry,
            seed=seed,
        )

    @override
    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: AnyFloat,
        b: AnyFloat,
        *,
        deg: int = 10,
    ) -> NumpyFloat:
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
            raise ValueError("LifetimeDistribution does not expect additional arguments in model_args")
        return super().fit(time, event=event, entry=entry, optimizer_options=optimizer_options)

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
            raise ValueError("LifetimeDistribution does not expect additional arguments in model_args")
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

    def __init__(self, rate: float | None = None):
        super().__init__(rate=rate)

    @property
    def rate(self):  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params["rate"]

    @override
    def hf(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def chf(self, time: AnyFloat) -> NumpyFloat:
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
        return np.asarray(self.rate, dtype=np.float64) * time

    @override
    def mean(self) -> NumpyFloat:
        """
        The mean of the distribution.

        Returns
        -------
        np.float64
        """
        return 1 / np.asarray(self.rate)

    @override
    def var(self) -> NumpyFloat:
        """
        The variance of the distribution.

        Returns
        -------
        np.float64
        """
        return 1 / np.asarray(self.rate) ** 2

    @override
    def mrl(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def ichf(self, cumulative_hazard_rate: AnyFloat) -> NumpyFloat:
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

    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_hf(self, time: AnyFloat, *, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
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
            jac = np.expand_dims(np.ones_like(time, dtype=np.float64), axis=0).copy()
        else:
            jac = np.array([1], dtype=np.float64)
        if not asarray:
            return np.unstack(jac)
        return jac

    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_chf(self, time: AnyFloat, *, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
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
            jac = np.expand_dims(time, axis=0).copy().astype(np.float64)
        else:
            jac = np.array([time], dtype=np.float64)
        if not asarray:
            return np.unstack(jac)
        return jac

    @override
    def dhf(self, time: AnyFloat) -> NumpyFloat:
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

    def __init__(self, shape: float | None = None, rate: float | None = None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self) -> float:  # optional but better for clarity and type checking
        """Get the current shape value.

        Returns
        -------
        float
        """
        return self._params["shape"]

    @property
    def rate(self) -> float:  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params["rate"]

    @override
    def hf(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def chf(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def mean(self) -> NumpyFloat:
        """
        The mean of the distribution.

        Returns
        -------
        np.float64
        """
        return gamma(1 + 1 / self.shape) / self.rate

    @override
    def var(self) -> NumpyFloat:
        """
        The variance of the distribution.

        Returns
        -------
        np.float64
        """
        return gamma(1 + 2 / self.shape) / self.rate**2 - self.mean() ** 2

    @override
    def mrl(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def ichf(self, cumulative_hazard_rate: AnyFloat) -> NumpyFloat:
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

    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_hf(self, time: AnyFloat, *, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
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

    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_chf(self, time: AnyFloat, *, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
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

    @override
    def dhf(self, time: AnyFloat) -> NumpyFloat:
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

    def __init__(self, shape: float | None = None, rate: float | None = None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self) -> float:  # optional but better for clarity and type checking
        """Get the current shape value.

        Returns
        -------
        float
        """
        return self._params["shape"]

    @property
    def rate(self) -> float:  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params["rate"]

    @override
    def hf(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def chf(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def mean(self) -> NumpyFloat:
        """
        The mean of the distribution.

        Returns
        -------
        np.float64
        """
        return np.exp(self.shape) * exp1(self.shape) / self.rate

    @override
    def var(self) -> NumpyFloat:
        """
        The variance of the distribution.

        Returns
        -------
        np.float64
        """
        return super().var()

    @override
    def mrl(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def ichf(self, cumulative_hazard_rate: AnyFloat) -> NumpyFloat:
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

    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_hf(self, time: AnyFloat, *, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
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

    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_chf(self, time: AnyFloat, *, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
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

    @override
    def dhf(self, time: AnyFloat) -> NumpyFloat:
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

    def __init__(self, shape: float | None = None, rate: float | None = None):
        super().__init__(shape=shape, rate=rate)

    def _uppergamma(self, x: AnyFloat) -> NumpyFloat:
        x = np.asarray(x, dtype=np.float64)
        return gammaincc(self.shape, x) * gamma(self.shape)

    def _jac_uppergamma_shape(self, x: AnyFloat) -> NumpyFloat:
        return laguerre_quadrature(lambda s: np.log(s) * s ** (self.shape - 1), x, deg=100)

    @property
    def shape(self) -> float:  # optional but better for clarity and type checking
        """Get the current shape value.

        Returns
        -------
        float
        """
        return self._params["shape"]

    @property
    def rate(self) -> float:  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params["rate"]

    @override
    def hf(self, time: AnyFloat) -> NumpyFloat:
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
        x = np.asarray(self.rate * time, dtype=np.float64)
        return self.rate * x ** (self.shape - 1) * np.exp(-x) / self._uppergamma(x)

    @override
    def chf(self, time: AnyFloat) -> NumpyFloat:
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
        x = np.asarray(self.rate * time, dtype=np.float64)
        return np.log(gamma(self.shape)) - np.log(self._uppergamma(x))

    @override
    def mean(self) -> NumpyFloat:
        """
        The mean of the distribution.

        Returns
        -------
        np.float64
        """
        return np.asarray(self.shape / self.rate)

    @override
    def var(self) -> NumpyFloat:
        """
        The variance of the distribution.

        Returns
        -------
        np.float64
        """
        return np.asarray(self.shape / (self.rate**2))

    @override
    def ichf(self, cumulative_hazard_rate: AnyFloat) -> NumpyFloat:
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
        return 1 / self.rate * np.asarray(gammainccinv(self.shape, np.exp(-cumulative_hazard_rate)), dtype=np.float64)

    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_hf(self, time: AnyFloat, *, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
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

    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_chf(self, time: AnyFloat, *, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
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

    @override
    def dhf(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def mrl(self, time: AnyFloat) -> NumpyFloat:
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

    def __init__(self, shape: float | None = None, rate: float | None = None):
        super().__init__(shape=shape, rate=rate)

    @property
    def shape(self) -> float:  # optional but better for clarity and type checking
        """Get the current shape value.

        Returns
        -------
        float
        """
        return self._params["shape"]

    @property
    def rate(self) -> float:  # optional but better for clarity and type checking
        """Get the current rate value.

        Returns
        -------
        float
        """
        return self._params["rate"]

    @override
    def hf(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def chf(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def mean(self) -> NumpyFloat:
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

    @override
    def var(self) -> NumpyFloat:
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

    @override
    def ichf(self, cumulative_hazard_rate: AnyFloat) -> NumpyFloat:
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

    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_hf(self, time: AnyFloat, *, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
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

    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_chf(self, time: AnyFloat, *, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
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

    @override
    def dhf(self, time: AnyFloat) -> NumpyFloat:
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

    @override
    def mrl(self, time: AnyFloat) -> NumpyFloat:
        return super().mrl(time)


Ts = TypeVarTuple("Ts")


@final
class EquilibriumDistribution(ParametricLifetimeModel[*Ts]):
    r"""Equilibrium distribution.

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
        return legendre_quadrature(lambda x: self.baseline.sf(x, *args), 0, time) / self.baseline.mean(*args)

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
    r"""Series structure of n identical and independent components.

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

    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        n: AnyInt,
        *args: *Ts,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        n: AnyInt,
        *args: *Ts,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_chf(
        self, time: AnyFloat, n: AnyInt, *args: *Ts, asarray: bool = True
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_chf(
        self, time: AnyFloat, n: AnyInt, *args: *Ts, asarray: bool = True
    ) -> tuple[NumpyFloat, ...] | NumpyFloat:
        return n * self.baseline.jac_chf(time, *args, asarray=asarray)

    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        n: AnyInt,
        *args: *Ts,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        n: AnyInt,
        *args: *Ts,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_hf(
        self, time: AnyFloat, n: AnyInt, *args: *Ts, asarray: bool = True
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_hf(
        self, time: AnyFloat, n: AnyInt, *args: *Ts, asarray: bool = True
    ) -> tuple[NumpyFloat, ...] | NumpyFloat:
        return n * self.baseline.jac_chf(time, *args, asarray=asarray)

    @overload
    def jac_sf(
        self,
        time: AnyFloat,
        n: AnyInt,
        *args: *Ts,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_sf(
        self,
        time: AnyFloat,
        n: AnyInt,
        *args: *Ts,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_sf(
        self, time: AnyFloat, n: AnyInt, *args: *Ts, asarray: bool = True
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_sf(
        self, time: AnyFloat, n: AnyInt, *args: *Ts, asarray: bool = False
    ) -> tuple[NumpyFloat, ...] | NumpyFloat:
        jac_chf, sf = (
            self.jac_chf(time, n, *args, asarray=True),
            self.sf(time, n, *args),
        )
        jac = -jac_chf * sf
        if not asarray:
            return np.unstack(jac)
        return jac

    @overload
    def jac_cdf(
        self,
        time: AnyFloat,
        n: AnyInt,
        *args: *Ts,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_cdf(
        self,
        time: AnyFloat,
        n: AnyInt,
        *args: *Ts,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_cdf(
        self, time: AnyFloat, n: AnyInt, *args: *Ts, asarray: bool = True
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    def jac_cdf(
        self, time: AnyFloat, n: AnyInt, *args: *Ts, asarray: bool = False
    ) -> tuple[NumpyFloat, ...] | NumpyFloat:
        jac = -self.jac_sf(time, n, *args, asarray=True)
        if not asarray:
            return np.unstack(jac)
        return jac

    @overload
    def jac_pdf(
        self,
        time: AnyFloat,
        n: AnyInt,
        *args: *Ts,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_pdf(
        self,
        time: AnyFloat,
        n: AnyInt,
        *args: *Ts,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_pdf(
        self, time: AnyFloat, n: AnyInt, *args: *Ts, asarray: bool = True
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_pdf(
        self, time: AnyFloat, n: AnyInt, *args: *Ts, asarray: bool = False
    ) -> tuple[NumpyFloat, ...] | NumpyFloat:
        jac_hf, hf = self.jac_hf(time, n, *args, asarray=True), self.hf(time, n, *args)
        jac_sf, sf = self.jac_sf(time, n, *args, asarray=True), self.sf(time, n, *args)
        jac = jac_hf * sf + jac_sf * hf
        if not asarray:
            return np.unstack(jac)
        return jac

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
            raise ValueError("MinimumDistribution expects at least one additional argument in model_args")
        return super().fit(time, model_args=model_args, event=event, entry=entry, optimizer_options=optimizer_options)

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
            raise ValueError("MinimumDistribution expects at least one additional argument in model_args")
        return super().fit_from_interval_censored_lifetimes(
            time_inf, time_sup, model_args=model_args, entry=entry, optimizer_options=optimizer_options
        )
