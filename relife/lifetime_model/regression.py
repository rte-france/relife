"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVarTuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from ._base import LifetimeRegression

if TYPE_CHECKING:
    from ._structural_type import FittableParametricLifetimeModel

Args = TypeVarTuple("Args")


# type ParametricLifetimeModel[float|NDArray[np.float64], *Ts]
class ProportionalHazard(LifetimeRegression[*Args]):
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
    baseline : :py:class:`~relife.model.protocols.FittableLifetimeDistribution`
        Any parametric_model lifetime model to serve as the baseline.
    coef : tuple of floats (values can be None), optional
        Coefficients values of the covariate effects.


    Attributes
    ----------
    params : np.ndarray
        The model parameters (both baseline parameters and covariate effects parameters).
    params_names : np.ndarray
        The model parameters (both baseline parameters and covariate effects parameters).
    baseline : :py:class:`~relife.model.protocols.FittableLifetimeDistribution`
        The regression baseline model.
    covar_effect : :py:class:`~relife.model.regression.CovarEffect`
        The regression covariate effect.


    References
    ----------
    .. [1] Sun, J. (2006). The statistical analysis of interval-censored failure
        time data (Vol. 3, No. 1). New York: springer.

    See Also
    --------
    regression.AFT : Accelerated Failure Time regression.

    """

    def hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.hf(time, *args)

    def chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        # (m,)
        return self.covar_effect.g(covar) * self.baseline.chf(time, *args)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        return self.baseline.ichf(
            cumulative_hazard_rate / self.covar_effect.g(covar), *args
        )

    def dhf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.dhf(time, *args)

    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
        asarray : bool = False,
    )  -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:

        time = np.atleast_2d(np.asarray(time)) # (m, n)
        covar = np.atleast_2d(np.asarray(covar)) # (m, nb_coef)

        match [time.shape[0], covar.shape[0]]:
            case [1, _]:
                time = np.repeat(time,covar.shape[0], axis=0)
            case [_, 1]:
                covar = np.repeat(covar, time.shape[0], axis=0)
            case [m1, m2] if m1 != m2:
                raise ValueError(f"Incompatible time and covar. time has {m1} nb_assets but covar has {m2} nb_assets")
        m, n = time.shape
        # covar : (m, nb_coef)

        g = self.covar_effect.g(covar) # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True) # (nb_coef, m, 1)

        baseline_hf = self.baseline.hf(time, *args) # (m, n)
        # p == baseline.nb_params
        baseline_jac_hf = self.baseline.jac_hf(time, *args, asarray=True) # (p, m, n)
        jac_g = np.repeat(jac_g, n, axis=-1) # (nb_coef, m, n)

        jac = np.concatenate((
            baseline_hf * jac_g,  #  (nb_coef, m, n)
            g * baseline_jac_hf, # (p, m, n)
        ), axis=0) # (p + nb_coef, m, n)

        match [m, n]:
            case [1, 1]:
                jac = jac.reshape(self.nb_params) # (p + nb_coef,)
            case [1, _]:
                jac = jac.reshape(self.nb_params, n)  #  (p + nb_coef, n)

        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
        asarray : bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:

        time = np.atleast_2d(np.asarray(time)) # (m, n)
        covar = np.atleast_2d(np.asarray(covar)) # (m, nb_coef)

        match [time.shape[0], covar.shape[0]]:
            case [1, _]:
                time = np.repeat(time,covar.shape[0], axis=0)
            case [_, 1]:
                covar = np.repeat(covar, time.shape[0], axis=0)
            case [m1, m2] if m1 != m2:
                raise ValueError(f"Incompatible time and covar. time has {m1} nb_assets but covar has {m2} nb_assets")
        m, n = time.shape
        # covar : (m, nb_coef)

        g = self.covar_effect.g(covar) # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True) # (nb_coef, m, 1)
        baseline_chf = self.baseline.chf(time, *args) # (m, n)
        #  p == baseline.nb_params
        baseline_jac_chf = self.baseline.jac_chf(time, *args, asarray=True) # (p, m, n)
        jac_g = np.repeat(jac_g, n, axis=-1) # (nb_coef, m, n)

        jac = np.concatenate((
            baseline_chf * jac_g,  #  (nb_coef, m, n)
            g * baseline_jac_chf, # (p, m, n)
        ), axis = 0) # (p + c, m, n)

        match [m, n]:
            case [1, 1]:
                jac = jac.reshape(self.nb_params) # (p + nb_coef,)
            case [1, _]:
                jac = jac.reshape(self.nb_params, n)  #  (p + nb_coef, n)

        if not asarray:
            return np.unstack(jac)
        return jac


class AcceleratedFailureTime(LifetimeRegression[*Args]):
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
    baseline : :py:class:`~relife.model.protocols.FittableLifetimeDistribution`
        Any parametric_model lifetime model to serve as the baseline.
    coef : tuple of floats (values can be None), optional
        Coefficients values of the covariate effects.

    Attributes
    ----------
    params : np.ndarray
        The model parameters (both baseline parameters and covariate effects parameters).
    params_names : np.ndarray
        The model parameters (both baseline parameters and covariate effects parameters).
    baseline : :py:class:`~relife.model.protocols.FittableLifetimeDistribution`
        The regression baseline model.
    covar_effect : :py:class:`~relife.model.regression.CovarEffect`
        The regression covariate effect.

    References
    ----------
    .. [1] Kalbfleisch, J. D., & Prentice, R. L. (2011). The statistical
        analysis of failure time data. John Wiley & Sons.

    See Also
    --------
    regression.ProportionalHazard : proportional hazard regression
    """

    def hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.hf(t0, *args) / self.covar_effect.g(covar)

    def chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.chf(t0, *args)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.ichf(
            cumulative_hazard_rate, *args
        )

    def dhf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
    ) -> np.float64 | NDArray[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.dhf(t0, *args) / self.covar_effect.g(covar) ** 2


    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
        asarray : bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:

        time = np.atleast_2d(np.asarray(time)) # (m, n)
        covar = np.atleast_2d(np.asarray(covar)) # (m, nb_coef)

        match [time.shape[0], covar.shape[0]]:
            case [1, _]:
                time = np.repeat(time,covar.shape[0], axis=0)
            case [_, 1]:
                covar = np.repeat(covar, time.shape[0], axis=0)
            case [m1, m2] if m1 != m2:
                raise ValueError(f"Incompatible time and covar. time has {m1} nb_assets but covar has {m2} nb_assets")
        m, n = time.shape
        # covar : (m, nb_coef)

        g = self.covar_effect.g(covar) # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True) # (nb_coef, m, 1)
        t0 = time / g # (m, n)
        # p == baseline.nb_params
        baseline_jac_hf_t0 = self.baseline.jac_hf(t0, *args, asarray=True) # (p, m, n)
        baseline_hf_t0 = self.baseline.hf(t0, *args) # (m, n)
        baseline_dhf_t0 = self.baseline.dhf(t0, *args) # (m, n)
        jac_g = np.repeat(jac_g, n, axis=-1) # (nb_coef, m, n)

        jac = np.concatenate((
            -jac_g / g ** 2 * (baseline_hf_t0 + t0 * baseline_dhf_t0),  #  (nb_coef, m, n)
            baseline_jac_hf_t0 / g, # (p, m, n)
        ), axis=0) # (p + c, m, n)

        match [m, n]:
            case [1, 1]:
                jac = jac.reshape(self.nb_params) # (p + nb_coef,)
            case [1, _]:
                jac = jac.reshape(self.nb_params, n)  #  (p + nb_coef, n)

        if not asarray:
            return np.unstack(jac)
        return jac


    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: *Args,
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64, ...] | tuple[NDArray[np.float64], ...]:

        time = np.atleast_2d(np.asarray(time)) # (m, n)
        covar = np.atleast_2d(np.asarray(covar)) # (m, nb_coef)

        match [time.shape[0], covar.shape[0]]:
            case [1, _]:
                time = np.repeat(time,covar.shape[0], axis=0)
            case [_, 1]:
                covar = np.repeat(covar, time.shape[0], axis=0)
            case [m1, m2] if m1 != m2:
                raise ValueError(f"Incompatible time and covar. time has {m1} nb_assets but covar has {m2} nb_assets")
        m, n = time.shape
        # covar : (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True) # (nb_coef, m, 1)
        t0 = time / g  #  (m, n)
        # p == baseline.nb_params
        baseline_jac_chf_t0 = self.baseline.jac_chf(t0, *args, asarray=True) # (p, m, n)
        baseline_hf_t0 = self.baseline.hf(t0, *args)  #  (m, n)
        jac_g = np.repeat(jac_g, n, axis=-1) # (nb_coef, m, n)

        jac = np.concatenate((
            -jac_g / g * t0 * baseline_hf_t0,  #  (nb_coef, m, n)
            baseline_jac_chf_t0, # (p, m, n)
        ),axis=0) # (p + nb_coef, m, n)

        match [m, n]:
            case [1, 1]:
                jac = jac.reshape(self.nb_params) # (p + nb_coef,)
            case [1, _]:
                jac = jac.reshape(self.nb_params, n)  #  (p + nb_coef, n)

        if not asarray:
            return np.unstack(jac)
        return jac



TIME_BASE_DOCSTRING = """
{name}.

Parameters
----------
time : float or np.ndarray
    Elapsed time value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
covar : np.ndarray
    Covariate values. The ndarray must be broadcastable with ``time``.
*args : variable number of np.ndarray
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
*args : variable number of np.ndarray
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
*args : variable number of np.ndarray
    Any variables needed to compute the function. Those variables must be
    broadcastable with ``time``. They may exist and result from method chaining due to nested class instantiation.

Returns
-------
np.ndarray
    {name} values.
"""


for class_obj in (AcceleratedFailureTime, ProportionalHazard):
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
