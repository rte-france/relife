"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, NewType, TypeVarTuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds
from typing_extensions import override

from relife.data import LifetimeData
from relife.distributions.abc import (
    SurvivalABC,
)
from relife.distributions.parameters import Parametric
from relife.distributions.protocols import (
    FittableLifetimeDistribution,
    LifetimeDistribution,
)
from relife.distributions.univariates import UnivariateRegression
from relife.likelihoods.mle import maximum_likelihood_estimation, FittingResults

Z = TypeVarTuple("Z")
T = NewType("T", NDArray[np.floating] | NDArray[np.integer] | float | int)
Covar = NewType("Covar", NDArray[np.floating] | NDArray[np.integer] | float | int)


class CovarEffect(Parametric):
    """
    Covariates effect.

    Parameters
    ----------
    coef : tuple of float or tuple of None, optional
        Coefficients of the covariates effect. Values can be None.
    """

    def __init__(self, coef: tuple[float, ...] | tuple[None] = (None,)):
        super().__init__()
        self.new_params(**{f"coef_{i}": v for i, v in enumerate(coef)})

    def g(self, covar: Covar) -> NDArray[np.float64]:
        """
        Compute the covariates effect.

        Parameters
        ----------
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values. Should have shape (k, ) or (m, k) where m is
            the number of assets and k is the number of covariates.

        Returns
        -------
        np.ndarray
            The covariate effect values, with shape (1,) or (m, 1).

        Raises
        ------
        ValueError
            If the number of covariates does not match the number of parameters.
        """
        if covar.shape[-1] != self.nb_params:
            raise ValueError(
                f"Invalid number of covar : expected {self.nb_params}, got {covar.shape[-1]}"
            )
        return np.exp(np.sum(self.params * covar, axis=1, keepdims=True))

    def jac_g(self, covar: Covar) -> NDArray[np.float64]:
        """
        Compute the Jacobian of the covariates effect.

        Parameters
        ----------
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values. Should have shape (k, ) or (m, k) where m is
            the number of assets and k is the number of covariates.

        Returns
        -------
        np.ndarray of shape (nb_params, ) or (m, nb_params)
            The values of the Jacobian (eventually for m assets).
        """
        return covar * self.g(covar)


# type is ParametricLifetimeModel[Covar, *Z] or LifetimeModel[Covar, *Z]
class Regression(Parametric, SurvivalABC[Covar, *Z], ABC):
    """
    Base class for regression distributions.
    """

    baseline: FittableLifetimeDistribution[*Z]
    covar_effect: CovarEffect

    def __init__(
        self,
        baseline: FittableLifetimeDistribution[*Z],
        coef: tuple[float, ...] | tuple[None] = (None,),
    ):
        super().__init__()
        if not isinstance(baseline, FittableLifetimeDistribution):
            raise ValueError(
                "Invalid baseline : must be FittableLifetimeDistribution object."
            )
        self.compose_with(
            covar_effect=CovarEffect(coef),
            baseline=baseline,
        )

    def init_params(
        self,
        lifetime_data: LifetimeData,
        covar: Covar,
        *z: *Z,
    ) -> None:
        """
        Initialize parameters for the regression core.

        Parameters
        ----------
        lifetime_data : LifetimeData
            The lifetime data used to initialize the baseline core.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values. Should have shape (k, ) or (m, k) where m is the number of assets and k is the number of covariates.
        *z : variable number of arguments
            Any additional arguments needed by the baseline core.
        """
        self.covar_effect.new_params(
            **{f"coef_{i}": 0.0 for i in range(covar.shape[-1])}
        )
        self.baseline.init_params(lifetime_data, *z)

    @property
    def params_bounds(self) -> Bounds:
        lb = np.concatenate(
            (
                np.full(self.covar_effect.nb_params, -np.inf),
                self.baseline.params_bounds.lb,
            )
        )
        ub = np.concatenate(
            (
                np.full(self.covar_effect.nb_params, np.inf),
                self.baseline.params_bounds.ub,
            )
        )
        return Bounds(lb, ub)

    @override
    def sf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return super().sf(time, covar, *z)

    @override
    def isf(
        self,
        probability: float | NDArray[np.float64],
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        """Inverse survival function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        covar : np.ndarray
            Covariate values. The ndarray must be broadcastable with ``time``.
        *z : variable number of np.ndarray
            Any variables needed to compute the function. Those variables must be
            broadcastable with ``time``. They may exist and result from method chaining due to nested class instantiation.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Time values corresponding to the given survival probabilities.
        """
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, covar, *z)

    @override
    def cdf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return super().cdf(time, covar, *z)

    def pdf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return super().pdf(time, covar, *z)

    @override
    def ppf(
        self,
        probability: float | NDArray[np.float64],
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return super().ppf(probability, covar, *z)

    @override
    def mrl(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return super().mrl(time, covar, *z)

    @override
    def rvs(
        self,
        covar: Covar,
        *z: *Z,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ):
        """
        Random variable sampling.

        Parameters
        ----------
        covar : np.ndarray
            Covariate values. Shapes can be ``(n_values,)`` or ``(n_assets, n_values)``.
        *z : variable number of np.ndarray
            Any variables needed to compute the function. Those variables must be
            broadcastable with ``covar``. They may exist and result from method chaining due to nested class instantiation.
        size : int, default 1
            Size of the sample.
        seed : int, default None
            Random seed.

        Returns
        -------
        np.ndarray
            Sample of random lifetimes.
        """
        return super().rvs(covar, *z, size=size, seed=seed)

    @override
    def mean(self, covar: Covar, *z: *Z) -> NDArray[np.float64]:
        return super().mean(covar, *z)

    @override
    def var(self, covar: Covar, *z: *Z) -> NDArray[np.float64]:
        return super().var(covar, *z)

    @override
    def median(self, covar: Covar, *z: *Z) -> NDArray[np.float64]:
        return super().median(covar, *z)

    @abstractmethod
    def jac_hf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def jac_chf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def dhf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]: ...

    def jac_sf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return -self.jac_chf(time, covar, *z) * self.sf(time, covar, *z)

    def jac_cdf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return -self.jac_sf(time, covar, *z)

    def jac_pdf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return self.jac_hf(time, covar, *z) * self.sf(time, covar, *z) + self.jac_sf(
            time, covar, *z
        ) * self.hf(time, covar, *z)

    @override
    def freeze_zvariables(
        self, covar: Covar, *z: *Z
    ) -> Union[UnivariateRegression[*Z], LifetimeDistribution[()]]:
        covar = np.atleast_2d(covar)
        return UnivariateRegression(self, *(covar, *z))

    def fit(
        self,
        time: NDArray[np.float64],
        covar: Covar,
        /,
        *z: *Z,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> FittingResults:
        # if update to 3.12 : maximum_likelihood_estimation[Covar, *Z](...), generic functions
        fitting_results = maximum_likelihood_estimation(
            self,
            time,
            covar,
            *z,
            event=event,
            entry=entry,
            departure=departure,
            **kwargs,
        )
        self.params = fitting_results.params
        return fitting_results


class ProportionalHazard(Regression[*Z]):
    r"""
    Proportional Hazard regression core.

    The cumulative hazard function :math:`H` is linked to the multiplier
    function :math:`g` by the relation:

    .. math::

        H(t, x) = g(\beta, x) H_0(t) = e^{\beta \cdot x} H_0(t)

    where :math:`x` is a vector of covariates, :math:`\beta` is the coefficient
    vector of the effect of covariates, :math:`H_0` is the baseline cumulative
    hazard function [1]_.

    |

    Parameters
    ----------
    baseline : :py:class:`~relife.models.protocols.FittableLifetimeDistribution`
        Any parametric lifetime model to serve as the baseline.
    coef : tuple of floats (values can be None), optional
        Coefficients values of the covariate effects.


    Attributes
    ----------
    params : np.ndarray
        The model parameters (both baseline parameters and covariate effects parameters).
    params_names : np.ndarray
        The model parameters (both baseline parameters and covariate effects parameters).
    baseline : :py:class:`~relife.models.protocols.FittableLifetimeDistribution`
        The regression baseline model.
    covar_effect : :py:class:`~relife.distributions.regression.CovarEffect`
        The regression covariate effect.


    References
    ----------
    .. [1] Sun, J. (2006). The statistical analysis of interval-censored failure
        time data (Vol. 3, No. 1). New York: springer.

    See Also
    --------
    regression.AFT : Accelerated Failure Time regression.

    """

    def __init__(
        self,
        baseline: FittableLifetimeDistribution[*Z],
        coef: tuple[float, ...] | tuple[None] = (None,),
    ):
        super().__init__(baseline, coef)

    def hf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.hf(time, *z)

    def chf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.chf(time, *z)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return self.baseline.ichf(
            cumulative_hazard_rate / self.covar_effect.g(covar), *z
        )

    def jac_hf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        if hasattr(self.baseline, "jac_hf"):
            return np.column_stack(
                (
                    self.covar_effect.jac_g(covar) * self.baseline.hf(time, *z),
                    self.covar_effect.g(covar) * self.baseline.jac_hf(time, *z),
                )
            )
        raise AttributeError

    def jac_chf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        if hasattr(self.baseline, "jac_chf"):
            return np.column_stack(
                (
                    self.covar_effect.jac_g(covar) * self.baseline.chf(time, *z),
                    self.covar_effect.g(covar) * self.baseline.jac_chf(time, *z),
                )
            )
        raise AttributeError

    def dhf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        if hasattr(self.baseline, "dhf"):
            return self.covar_effect.g(covar) * self.baseline.dhf(time, *z)
        raise AttributeError


class AFT(Regression[*Z]):
    r"""
    Accelerated failure time regression.

    The cumulative hazard function :math:`H` is linked to the multiplier
    function :math:`g` by the relation:

    .. math::

        H(t, x) = H_0\left(\dfrac{t}{g(\beta, x)}\right) = H_0(t e^{- \beta
        \cdot x})

    where :math:`x` is a vector of covariates, :math:`\beta` is the coefficient
    vector of the effect of covariates, :math:`H_0` is the baseline cumulative
    hazard function [1]_.

    |

    Parameters
    ----------
    baseline : :py:class:`~relife.models.protocols.FittableLifetimeDistribution`
        Any parametric lifetime model to serve as the baseline.
    coef : tuple of floats (values can be None), optional
        Coefficients values of the covariate effects.

    Attributes
    ----------
    params : np.ndarray
        The model parameters (both baseline parameters and covariate effects parameters).
    params_names : np.ndarray
        The model parameters (both baseline parameters and covariate effects parameters).
    baseline : :py:class:`~relife.models.protocols.FittableLifetimeDistribution`
        The regression baseline model.
    covar_effect : :py:class:`~relife.distributions.regression.CovarEffect`
        The regression covariate effect.

    References
    ----------
    .. [1] Kalbfleisch, J. D., & Prentice, R. L. (2011). The statistical
        analysis of failure time data. John Wiley & Sons.

    See Also
    --------
    regression.ProportionalHazard : proportional hazard regression
    """

    def __init__(
        self,
        baseline: FittableLifetimeDistribution[*Z],
        coef: tuple[float, ...] | tuple[None] = (None,),
    ):
        super().__init__(baseline, coef)

    def hf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.hf(t0, *z) / self.covar_effect.g(covar)

    def chf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.chf(t0, *z)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.ichf(
            cumulative_hazard_rate, *z
        )

    def jac_hf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        if hasattr(self.baseline, "jac_hf") and hasattr(self.baseline, "dhf"):
            t0 = time / self.covar_effect.g(covar)
            return np.column_stack(
                (
                    -self.covar_effect.jac_g(covar)
                    / self.covar_effect.g(covar) ** 2
                    * (self.baseline.hf(t0, *z) + t0 * self.baseline.dhf(t0, *z)),
                    self.baseline.jac_hf(t0, *z) / self.covar_effect.g(covar),
                )
            )
        raise AttributeError

    def jac_chf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        if hasattr(self.baseline, "jac_chf"):
            t0 = time / self.covar_effect.g(covar)
            return np.column_stack(
                (
                    -self.covar_effect.jac_g(covar)
                    / self.covar_effect.g(covar)
                    * t0
                    * self.baseline.hf(t0, *z),
                    self.baseline.jac_chf(t0, *z),
                )
            )
        raise AttributeError

    def dhf(
        self,
        time: T,
        covar: Covar,
        *z: *Z,
    ) -> NDArray[np.float64]:
        if hasattr(self.baseline, "dhf"):
            t0 = time / self.covar_effect.g(covar)
            return self.baseline.dhf(t0, *z) / self.covar_effect.g(covar) ** 2
        raise AttributeError


TIME_BASE_DOCSTRING = """
{name}.

Parameters
----------
time : float or np.ndarray
    Elapsed time value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
covar : np.ndarray
    Covariate values. The ndarray must be broadcastable with ``time``.
*z : variable number of np.ndarray
    Any variables needed to compute the function. Those variables must be
    broadcastable with ``time``. They may exist and result from method chaining due to nested class instantiation.

Returns
-------
np.ndarray
    Function values at each given time(s).
"""


ICHF_DOCSTRING = """
Inverse cumulative hazard function.

Parameters
----------
cumulative_hazard_rate : float or np.ndarray
    Cumulative hazard rate value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
covar : np.ndarray
    Covariate values. The ndarray must be broadcastable with ``time``.
*z : variable number of np.ndarray
    Any variables needed to compute the function.

Returns
-------
np.ndarray
    Function values at each given time(s).
"""

MOMENT_BASE_DOCSTRING = """
{name}.

Parameters
----------
covar : np.ndarray
    Covariate values. The ndarray must be broadcastable with ``time``.
*z : variable number of np.ndarray
    Any variables needed to compute the function. Those variables must be
    broadcastable with ``time``. They may exist and result from method chaining due to nested class instantiation.

Returns
-------
np.ndarray
    {name} values.
"""


for class_obj in (AFT, ProportionalHazard):
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
