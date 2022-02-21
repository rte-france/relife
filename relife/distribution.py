"""Parametric lifetime distribution."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Tuple
import numpy as np
from scipy.optimize import Bounds
from scipy.special import exp1, gamma, gammaincc, gammainccinv, digamma

from .data import LifetimeData
from .parametric import ParametricLifetimeModel
from .utils import MIN_POSITIVE_FLOAT, plot, shifted_laguerre


@dataclass
class ParametricLifetimeDistribution(ParametricLifetimeModel):
    """Generic class for parametric lifetime distribution."""

    @property
    def n_params(self) -> int:
        return len(fields(self))

    @property
    def _param_bounds(self) -> Bounds:
        # return Bounds(*[(MIN_POSITIVE_FLOAT, None)] * len(fields(self)))
        return Bounds(
            np.full(self.n_params, MIN_POSITIVE_FLOAT), np.full(self.n_params, np.inf)
        )

    def _init_params(self, data: LifetimeData) -> np.ndarray:
        params0 = np.ones(self.n_params)
        params0[-1] = 1 / np.median(data.time)
        return params0

    def _set_params(self, params: np.ndarray) -> None:
        for i, field in enumerate(fields(self)):
            setattr(self, field.name, params[i])

    def fit(
        self,
        time: np.ndarray,
        event: np.ndarray = None,
        entry: np.ndarray = None,
        args: np.ndarray = (),
        params0: np.ndarray = None,
        method: str = None,
        **kwargs,
    ) -> ParametricLifetimeDistribution:
        """Fit the parametric lifetime distribution to lifetime data.

        Parameters
        ----------
        time : 1D array
            Array of time-to-event or durations.
        event : 1D array, optional
            Array of event types coded as follows:

            - 0 if observation ends before the event has occurred (right censoring)
            - 1 if the event has occured
            - 2 if observation starts after the event has occurred (left censoring)

            by default the event has occured for each asset.
        entry : 1D array, optional
            Array of delayed entry times (left truncation),
            by default None.
        args : float or 2D array, optional
            Extra arguments required by the parametric lifetime model.
        params0 : 1D array, optional
            Initial guess, by default None.
        method : str, optional
            Type of solver (see scipy.optimize.minimize documentation), by
            default None.
        **kwargs: dict, optional
            Extra arguments to pass to the minimize method.

        Returns
        -------
        self
            Return the fitted distribution as the current object.
        """
        data = LifetimeData(time, event, entry, args)
        self._fit(data, params0, method=method, **kwargs)
        return self

    def plot(
        self,
        timeline: np.ndarray = None,
        args: Tuple[np.ndarray] = (),
        alpha_ci: float = 0.05,
        fname: str = "sf",
        **kwargs,
    ) -> None:
        r"""Plot functions of the distribution model.

        Parameters
        ----------
        timeline : 1D array, optional
            Timeline of the plot (x-axis), by default guessed by the millile.
        args : Tuple[ndarray], optional
            Extra arguments required by the parametric lifetime model, by
            default ().
        alpha_ci : float, optional
            :math:`\alpha`-value to define the :math:`100(1-\alpha)\%`
            confidence interval, by default 0.05 corresponding to the 95\%
            confidence interval. If set to None or if the model has not been
            fitted, no confidence interval is plotted.
        fname : str, optional
            Name of the function to be plotted, by default 'sf'. Should be one
            of:

            - 'sf': survival function,
            - 'cdf': cumulative distribution function,
            - 'chf': cumulative hazard function,
            - 'hf': hazard function,
            - 'pdf': probability density function.

        **kwargs : dict, optional
            Extra arguments to specify the plot properties (see
            matplotlib.pyplot.plot documentation).

        Raises
        ------
        ValueError
            If `fname` value is not among 'sf', 'cdf', 'chf', 'hf' or 'pdf'.
        """
        flist = ["sf", "cdf", "chf", "hf", "pdf"]
        if fname not in flist:
            raise ValueError(
                "Function name '{}' is not supported for plotting, `fname` must be in {}".format(
                    fname, flist
                )
            )
        if timeline is None:
            timeline = np.linspace(0, self.isf(1e-3), 200)
        f = getattr(self, "_" + fname)
        jac_f = getattr(self, "_jac_" + fname)
        y = f(self.params, timeline, *args)
        if alpha_ci is not None and hasattr(self, "result"):
            i0 = 0
            se = np.empty_like(timeline, float)
            if timeline[0] == 0:
                i0 = 1
                se[0] = 0
            se[i0:] = self.result.standard_error(
                jac_f(self.result.opt.x, timeline[i0:].reshape(-1, 1), *args)
            )
        else:
            se = None
        label = kwargs.pop("label", self.__class__.__name__)
        bounds = (0, 1) if fname in ["sf", "cdf"] else (0, np.inf)
        plot(timeline, y, se, alpha_ci, bounds=bounds, label=label, **kwargs)


@dataclass
class Exponential(ParametricLifetimeDistribution):
    r"""Exponential parametric lifetime distribution.

    The exponential distribution is a 1-parameter distribution with
    :math:`(\lambda)`. The probability density function is:

    .. math::

        f(t) = \lambda e^{-\lambda t}

    where:

        - :math:`\lambda > 0`, the rate parameter,
        - :math:`t\geq 0`, the operating time, age, cycles, etc.
    """

    rate: float = None  #: rate parameter (inverse of scale)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.rate])

    def _chf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        rate = params[0]
        return rate * t

    def _hf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        rate = params[0]
        return rate * np.ones_like(t)

    def _dhf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.zeros_like(t)

    def _jac_chf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.ones((t.size, 1)) * t

    def _jac_hf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.ones((t.size, 1))

    def _ichf(self, params: np.ndarray, v: np.ndarray) -> np.ndarray:
        rate = params[0]
        return v / rate

    def mean(self) -> np.ndarray:
        rate = self.params[0]
        return 1 / rate

    def var(self) -> np.ndarray:
        rate = self.params[0]
        return 1 / rate**2

    def mrl(self, t: np.ndarray) -> np.ndarray:
        rate = self.params[0]
        return 1 / rate * np.ones_like(t)


@dataclass
class Weibull(ParametricLifetimeDistribution):
    r"""Weibull parametric lifetime distribution.

    The Weibull distribution is a 2-parameter distribution with
    :math:`(c,\lambda)`. The probability density function is:

    .. math::

        f(t) = c \lambda^c t^{c-1} e^{-(\lambda t)^c}

    where:

        - :math:`c > 0`, the shape parameter,
        - :math:`\lambda > 0`, the rate parameter,
        - :math:`t\geq 0`, the operating time, age, cycles, etc.
    """

    c: float = None  #: shape parameter
    rate: float = None  #: rate parameter (inverse of scale)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.c, self.rate])

    def _chf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        return (rate * t) ** c

    def _hf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        return c * rate * (rate * t) ** (c - 1)

    def _dhf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        return c * (c - 1) * rate**2 * (rate * t) ** (c - 2)

    def _jac_chf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        return np.column_stack(
            (np.log(rate * t) * (rate * t) ** c, c * t * (rate * t) ** (c - 1))
        )

    def _jac_hf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        return np.column_stack(
            (
                rate * (rate * t) ** (c - 1) * (1 + c * np.log(rate * t)),
                c**2 * (rate * t) ** (c - 1),
            )
        )

    def _ichf(self, params: np.ndarray, v: np.ndarray) -> np.ndarray:
        c, rate = params
        return v ** (1 / c) / rate

    def mean(self) -> np.ndarray:
        c, rate = self.params
        return gamma(1 + 1 / c) / rate

    def mrl(self, t: np.ndarray) -> np.ndarray:
        c, rate = self.params
        return (
            gamma(1 / c) / (rate * c * self.sf(t)) * gammaincc(1 / c, (rate * t) ** c)
        )

    def var(self) -> np.ndarray:
        c, rate = self.params
        return gamma(1 + 2 / c) / rate**2 - self.mean() ** 2


@dataclass
class Gompertz(ParametricLifetimeDistribution):
    r"""Gompertz parametric lifetime distribution.

    The Gompertz distribution is a 2-parameter distribution with
    :math:`(c,\lambda)`. The probability density function is:

    .. math::

        f(t) = c \lambda e^{\lambda t} e^{ -c \left( e^{\lambda t}-1 \right) }

    where:

        - :math:`c > 0`, the shape parameter,
        - :math:`\lambda > 0`, the rate parameter,
        - :math:`t\geq 0`, the operating time, age, cycles, etc.
    """

    c: float = None  #: shape parameter
    rate: float = None  #: rate parameter (inverse of scale)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.c, self.rate])

    def _init_params(self, data: LifetimeData) -> np.ndarray:
        rate = np.pi / (np.sqrt(6) * np.std(data.time))
        c = np.exp(-rate * np.mean(data.time))
        return np.array([c, rate])

    def _chf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        return c * np.expm1(rate * t)

    def _hf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        return c * rate * np.exp(rate * t)

    def _dhf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        return c * (rate) ** 2 * np.exp(rate * t)

    def _jac_chf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        return np.column_stack((np.expm1(rate * t), c * t * np.exp(rate * t)))

    def _jac_hf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        return np.column_stack(
            (rate * np.exp(rate * t), c * np.exp(rate * t) * (1 + rate * t))
        )

    def _ichf(self, params: np.ndarray, v: np.ndarray) -> np.ndarray:
        c, rate = params
        return 1 / rate * np.log1p(v / c)

    def mean(self) -> np.ndarray:
        c, rate = self.params
        return np.exp(c) * exp1(c) / rate

    def mrl(self, t: np.ndarray) -> np.ndarray:
        c, rate = self.params
        z = c * np.exp(rate * t)
        return np.exp(z) * exp1(z) / rate


@dataclass
class Gamma(ParametricLifetimeDistribution):
    r"""Gamma parametric lifetime distribution.

    The Gamma distribution is a 2-parameter distribution with
    :math:`(c,\lambda)`. The probability density function is:

    .. math::

        f(t) = \frac{\lambda^c t^{c-1} e^{-\lambda t}}{\Gamma(c)}

    where:

        - :math:`c > 0`, the shape parameter,
        - :math:`\lambda > 0`, the rate parameter,
        - :math:`t\geq 0`, the operating time, age, cycles, etc.
    """

    c: float = None  # shape parameter
    rate: float = None  #: rate parameter (inverse of scale)

    @staticmethod
    def _uppergamma(c: np.ndarray, x: np.ndarray) -> np.ndarray:
        return gammaincc(c, x) * gamma(c)

    @staticmethod
    def _jac_uppergamma_c(c: np.ndarray, x: np.ndarray) -> np.ndarray:
        return shifted_laguerre(lambda s: np.log(s) * s ** (c - 1), x, ndim=np.ndim(x))

    @property
    def _default_hess_scheme(self):
        return "2-point"

    @property
    def params(self) -> np.ndarray:
        return np.array([self.c, self.rate])

    def _chf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        x = rate * t
        return np.log(gamma(c)) - np.log(self._uppergamma(c, x))

    def _hf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        x = rate * t
        return rate * x ** (c - 1) * np.exp(-x) / self._uppergamma(c, x)

    def _dhf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        return self._hf(params, t) * ((c - 1) / t - rate + self._hf(params, t))

    def _jac_chf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        x = rate * t
        return np.column_stack(
            (
                digamma(c) - self._jac_uppergamma_c(c, x) / self._uppergamma(c, x),
                x ** (c - 1) * t * np.exp(-x) / self._uppergamma(c, x),
            )
        )

    def _jac_hf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        x = rate * t
        return (
            x ** (c - 1)
            * np.exp(-x)
            / self._uppergamma(c, x) ** 2
            * np.column_stack(
                (
                    rate * np.log(x) * self._uppergamma(c, x)
                    - rate * self._jac_uppergamma_c(c, x),
                    (c - x) * self._uppergamma(c, x) + x**c * np.exp(-x),
                )
            )
        )

    def _ichf(self, params: np.ndarray, v: np.ndarray) -> np.ndarray:
        c, rate = params
        return 1 / rate * gammainccinv(c, np.exp(-v))

    def mean(self) -> np.ndarray:
        c, rate = self.params
        return c / rate

    def var(self, params: np.ndarray) -> np.ndarray:
        c, rate = params
        return c / (rate**2)


@dataclass
class LogLogistic(ParametricLifetimeDistribution):
    r"""Log-logistic parametric lifetime distribution.

    The Log-logistic distribution is defined as a 2-parameter distribution
    :math:`(c, \lambda)`. The probability density function is:

    .. math::

        f(t) = \frac{c \lambda^c t^{c-1}}{(1+(\lambda t)^{c})^2}

    where:

        - :math:`c > 0`, the shape parameter,
        - :math:`\lambda > 0`, the rate parameter,
        - :math:`t\geq 0`, the operating time, age, cycles, etc.
    """

    c: float = None  #: shape parameter
    rate: float = None  #: rate parameter (inverse of scale)

    @property
    def params(self) -> np.ndarray:
        return np.array([self.c, self.rate])

    def _chf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        x = rate * t
        return np.log(1 + x**c)

    def _hf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        x = rate * t
        return c * rate * x ** (c - 1) / (1 + x**c)

    def _dhf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        x = rate * t
        return c * rate**2 * x ** (c - 2) * (c - 1 - x**c) / (1 + x**c) ** 2

    def _jac_chf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        x = rate * t
        return np.column_stack(
            (
                (x**c / (1 + x**c)) * np.log(rate * t),
                (x**c / (1 + x**c)) * (c / rate),
            )
        )

    def _jac_hf(self, params: np.ndarray, t: np.ndarray) -> np.ndarray:
        c, rate = params
        x = rate * t
        return np.column_stack(
            (
                (rate * x ** (c - 1) / (1 + x**c) ** 2)
                * (1 + x**c + c * np.log(rate * t)),
                (rate * x ** (c - 1) / (1 + x**c) ** 2) * (c**2 / rate),
            )
        )

    def _ichf(self, params: np.ndarray, v: np.ndarray) -> np.ndarray:
        c, rate = params
        return ((np.exp(v) - 1) ** (1 / c)) / rate

    def mean(self) -> np.ndarray:
        c, rate = self.params
        b = np.pi / c
        if c > 1:
            return b / (rate * np.sin(b))
        else:
            raise ValueError(f"Expectancy only defined for c > 1: c = {c}")

    def var(self) -> np.ndarray:
        c, rate = self.params
        b = np.pi / c
        if c > 2:
            return (1 / rate**2) * (2 * b / np.sin(2 * b) - b**2 / (np.sin(b) ** 2))
        else:
            raise ValueError(f"Variance only defined for c > 2: c = {c}")


@dataclass
class MinimumDistribution(ParametricLifetimeDistribution):
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

    baseline: ParametricLifetimeDistribution  #: Underlying lifetime model of the components.

    @property
    def _default_hess_scheme(self) -> str:
        return self.baseline._default_hess_scheme

    @property
    def params(self) -> np.ndarray:
        return self.baseline.params

    @property
    def n_params(self) -> int:
        return self.baseline.n_params

    @property
    def _param_bounds(self) -> Bounds:
        return self.baseline._param_bounds

    def _set_params(self, params: np.ndarray) -> None:
        self.baseline._set_params(params)

    def _chf(
        self, params: np.ndarray, t: np.ndarray, n: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        return n * self.baseline._chf(params, t, *args)

    def _hf(
        self, params: np.ndarray, t: np.ndarray, n: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        return n * self.baseline._hf(params, t, *args)

    def _dhf(
        self, params: np.ndarray, t: np.ndarray, n: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        return n * self.baseline._dhf(params, t, *args)

    def _jac_chf(
        self, params: np.ndarray, t: np.ndarray, n: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        return n * self.baseline._jac_chf(params, t, *args)

    def _jac_hf(
        self, params: np.ndarray, t: np.ndarray, n: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        return n * self.baseline._jac_hf(params, t, *args)

    def _ichf(
        self, params: np.ndarray, v: np.ndarray, n: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        return self.baseline._ichf(params, v / n, *args)
