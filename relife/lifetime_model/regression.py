"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from ._base import LifetimeRegression


def broadcast_time_covar(
    time: float | NDArray[np.float64],
    covar: float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    time = np.atleast_2d(np.asarray(time))  #  (m, n)
    covar = np.atleast_2d(np.asarray(covar))  #  (m, nb_coef)
    match (time.shape[0], covar.shape[0]):
        case (1, _):
            time = np.repeat(time, covar.shape[0], axis=0)
        case (_, 1):
            covar = np.repeat(covar, time.shape[0], axis=0)
        case (m1, m2) if m1 != m2:
            raise ValueError(f"Incompatible time and covar. time has {m1} nb_assets but covar has {m2} nb_assets")
    return time, covar


def broadcast_time_covar_shapes(
    time_shape: tuple[()] | tuple[int] | tuple[int, int],
    covar_shape: tuple[()] | tuple[int] | tuple[int, int],
) -> tuple[()] | tuple[int] | tuple[int, int]:
    # time_shape : (), (n,) or (m, n)
    # covar_shape : (), (nb_coef,) or (m, nb_coef)
    match [time_shape, covar_shape]:
        case [(), ()] | [(), (_,)]:
            return ()
        case [(), (m, _)]:
            return m, 1
        case [(n,), ()] | [(n,), (_,)]:
            return (n,)
        case [(n,), (m, _)] | [(m, n), ()] | [(m, n), (_,)]:
            return m, n
        case [(mt, n), (mc, _)] if mt != mc:
            if mt != 1 and mc != 1:
                raise ValueError(f"Invalid time and covar : time got {mt} nb assets but covar got {mc} nb assets")
            return max(mt, mc), n
        case [(mt, n), (mc, _)] if mt == mc:
            return mt, n
        case _:
            raise ValueError(f"Invalid time or covar shape. Got {time_shape} and {covar_shape}")


class ProportionalHazard(LifetimeRegression):
    # noinspection PyUnresolvedReferences
    r"""
    Proportional Hazard regression.

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
    baseline : FittableParametricLifetimeModel
        Any lifetime model that can be fitted.
    coefficients : tuple of floats (values can be None), default is (None,)
        Coefficients values of the covariate effects.

    Attributes
    ----------
    baseline : FittableParametricLifetimeModel
        The regression baseline model (lifetime model).
    covar_effect : CovarEffect
        The regression covariate effect.
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.). If the model is not fitted, the value is None.
    coefficients
    nb_params
    params
    params_names
    plot


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
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.hf(time, *args)

    def chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        # (m,)
        return self.covar_effect.g(covar) * self.baseline.chf(time, *args)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        return self.baseline.ichf(cumulative_hazard_rate / self.covar_effect.g(covar), *args)

    def dhf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.dhf(time, *args)

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:

        time = np.asarray(time)  # (), (n,) or (m, n)
        covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        out_shape = broadcast_time_covar_shapes(time.shape, covar.shape)  # (), (n,) or (m, n)
        time, covar = broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True)  # (nb_coef, m, 1)

        baseline_hf = self.baseline.hf(time, *args)  # (m, n)
        # p == baseline.nb_params
        baseline_jac_hf = self.baseline.jac_hf(time, *args, asarray=True)  # (p, m, n)
        jac_g = np.repeat(jac_g, baseline_hf.shape[-1], axis=-1)  # (nb_coef, m, n) necessary to concatenate

        jac = np.concatenate(
            (
                baseline_hf * jac_g,  #  (nb_coef, m, n)
                g * baseline_jac_hf,  # (p, m, n)
            ),
            axis=0,
        )  # (p + nb_coef, m, n)

        jac = jac.reshape((self.nb_params,) + out_shape)
        if not asarray:
            return np.unstack(jac)
        return jac

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:

        time = np.asarray(time)  # (), (n,) or (m, n)
        covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        out_shape = broadcast_time_covar_shapes(time.shape, covar.shape)  # (), (n,) or (m, n)
        time, covar = broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True)  # (nb_coef, m, 1)
        baseline_chf = self.baseline.chf(time, *args)  # (m, n)
        #  p == baseline.nb_params
        baseline_jac_chf = self.baseline.jac_chf(time, *args, asarray=True)  # (p, m, n)
        jac_g = np.repeat(jac_g, baseline_chf.shape[-1], axis=-1)  # (nb_coef, m, n) necessary to concatenate

        jac = np.concatenate(
            (
                baseline_chf * jac_g,  #  (nb_coef, m, n)
                g * baseline_jac_chf,  # (p, m, n)
            ),
            axis=0,
        )  # (p + nb_coef, m, n)

        jac = jac.reshape((self.nb_params,) + out_shape)
        if not asarray:
            return np.unstack(jac)
        return jac

    @override
    def moment(
        self, n: int, covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        return super().moment(n, *(covar, *args))


class AcceleratedFailureTime(LifetimeRegression):
    # noinspection PyUnresolvedReferences
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
    baseline : FittableParametricLifetimeModel
        Any lifetime model that can be fitted.
    coefficients : tuple of floats (values can be None), default is (None,)
        Coefficients values of the covariate effects.

    Attributes
    ----------
    baseline : FittableParametricLifetimeModel
        The regression baseline model (lifetime model).
    covar_effect : CovarEffect
        The regression covariate effect.
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.). If the model is not fitted, the value is None.
    coefficients
    nb_params
    params
    params_names
    plot


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
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.hf(t0, *args) / self.covar_effect.g(covar)

    def chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.chf(t0, *args)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.ichf(cumulative_hazard_rate, *args)

    def dhf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
    ) -> np.float64 | NDArray[np.float64]:
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.dhf(t0, *args) / self.covar_effect.g(covar) ** 2

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:

        time = np.asarray(time)  # (), (n,) or (m, n)
        covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        out_shape = broadcast_time_covar_shapes(time.shape, covar.shape)  # (), (n,) or (m, n)
        time, covar = broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True)  # (nb_coef, m, 1)
        t0 = time / g  # (m, n)
        # p == baseline.nb_params
        baseline_jac_hf_t0 = self.baseline.jac_hf(t0, *args, asarray=True)  # (p, m, n)
        baseline_hf_t0 = self.baseline.hf(t0, *args)  # (m, n)
        baseline_dhf_t0 = self.baseline.dhf(t0, *args)  # (m, n)
        jac_g = np.repeat(jac_g, baseline_hf_t0.shape[-1], axis=-1)  # (nb_coef, m, n)

        jac = np.concatenate(
            (
                -jac_g / g**2 * (baseline_hf_t0 + t0 * baseline_dhf_t0),  # (nb_coef, m, n) necessary to concatenate
                baseline_jac_hf_t0 / g,  # (p, m, n)
            ),
            axis=0,
        )  # (p + nb_coef, m, n)

        jac = jac.reshape((self.nb_params,) + out_shape)
        if not asarray:
            return np.unstack(jac)
        return jac

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[False] = False,
    ) -> tuple[np.float64 | NDArray[np.float64], ...]: ...

    @overload
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: Literal[True] = True,
    ) -> np.float64 | NDArray[np.float64]: ...

    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: float | NDArray[np.float64],
        *args: float | NDArray[np.float64],
        asarray: bool = False,
    ) -> np.float64 | NDArray[np.float64] | tuple[np.float64 | NDArray[np.float64], ...]:

        time = np.asarray(time)  # (), (n,) or (m, n)
        covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        out_shape = broadcast_time_covar_shapes(time.shape, covar.shape)  # (), (n,) or (m, n)
        time, covar = broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True)  # (nb_coef, m, 1)
        t0 = time / g  #  (m, n)
        # p == baseline.nb_params
        baseline_jac_chf_t0 = self.baseline.jac_chf(t0, *args, asarray=True)  # (p, m, n)
        baseline_hf_t0 = self.baseline.hf(t0, *args)  #  (m, n)
        jac_g = np.repeat(jac_g, baseline_hf_t0.shape[-1], axis=-1)  # (nb_coef, m, n) necessary to concatenate

        jac = np.concatenate(
            (
                -jac_g / g * t0 * baseline_hf_t0,  #  (nb_coef, m, n)
                baseline_jac_chf_t0,  # (p, m, n)
            ),
            axis=0,
        )  # (p + nb_coef, m, n)

        jac = jac.reshape((self.nb_params,) + out_shape)
        if not asarray:
            return np.unstack(jac)
        return jac

    @override
    def moment(
        self, n: int, covar: float | NDArray[np.float64], *args: float | NDArray[np.float64]
    ) -> np.float64 | NDArray[np.float64]:
        return super().moment(n, *(covar, *args))


TIME_BASE_DOCSTRING = """
{name}.

Parameters
----------
time : float or np.ndarray
    Elapsed time value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
covar : float or np.ndarray
    Covariates values. float can only be valid if the regression has one coefficients. 
    Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)
*args : float or np.ndarray
    Additional arguments needed by the model.

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
    If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
covar : float or np.ndarray
    Covariates values. float can only be valid if the regression has one coefficients. 
    Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)
*args : float or np.ndarray
    Additional arguments needed by the model.
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

Parameters
----------
covar : float or np.ndarray
    Covariates values. float can only be valid if the regression has one coefficients. 
    Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.
*args : float or np.ndarray
    Additional arguments needed by the model.

Returns
-------
np.float64
    {name} value.
"""

ICHF_DOCSTRING = """
Inverse cumulative hazard function.

Parameters
----------
cumulative_hazard_rate : float or np.ndarray
    Cumulative hazard rate value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
covar : float or np.ndarray
    Covariates values. float can only be valid if the regression has one coefficients. 
    Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.
*args : float or np.ndarray
    Additional arguments needed by the model.

Returns
-------
np.float64 or np.ndarray
    Function values at each given cumulative hazard rate(s).
"""


PROBABILITY_BASE_DOCSTRING = """
{name}.

Parameters
----------
probability : float or np.ndarray
    Probability value(s) at which to compute the function.
    If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
covar : float or np.ndarray
    Covariates values. float can only be valid if the regression has one coefficients. 
    Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.
*args : float or np.ndarray
    Additional arguments needed by the model.

Returns
-------
np.float64 or np.ndarray
    Function values at each given probability value(s).
"""


for class_obj in (AcceleratedFailureTime, ProportionalHazard):
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
    covar : float or np.ndarray
        Covariates values. float can only be valid if the regression has one coefficients. 
        Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.
    *args : float or np.ndarray
        Additional arguments needed by the model.
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
        
    covar : float or np.ndarray
        Covariates values. float can only be valid if the regression has one coefficients. 
        Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.
    *args : float or np.ndarray
        Additional arguments needed by the model.

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
    covar : float or np.ndarray
        Covariates values. float can only be valid if the regression has one coefficients. 
        Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.
    *args : float or np.ndarray
        Additional arguments needed by the model.
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
