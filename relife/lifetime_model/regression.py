"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC

import numpy as np
from scipy.optimize import Bounds

from relife.base import FrozenParametricModel, ParametricModel

from ._base import FittableParametricLifetimeModel

__all__ = ["LifetimeRegression", "AcceleratedFailureTime", "ProportionalHazard"]


def _broadcast_time_covar(time, covar):
    time = np.atleast_2d(np.asarray(time))  #  (m, n)
    covar = np.atleast_2d(np.asarray(covar))  #  (m, nb_coef)
    match (time.shape[0], covar.shape[0]):
        case (1, _):
            time = np.repeat(time, covar.shape[0], axis=0)
        case (_, 1):
            covar = np.repeat(covar, time.shape[0], axis=0)
        case (m1, m2) if m1 != m2:
            raise ValueError(
                f"Incompatible time and covar. time has {m1} nb_assets but covar has {m2} nb_assets"
            )
    return time, covar

def _broadcast_time_covar_shapes(time_shape, covar_shape):
    """
    time_shape : (), (n,) or (m, n)
    covar_shape : (), (nb_coef,) or (m, nb_coef)
    """
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
                raise ValueError(
                    f"Invalid time and covar : time got {mt} nb assets but covar got {mc} nb assets"
                )
            return max(mt, mc), n
        case [(mt, n), (mc, _)] if mt == mc:
            return mt, n
        case _:
            raise ValueError(
                f"Invalid time or covar shape. Got {time_shape} and {covar_shape}"
            )


class _CovarEffect(ParametricModel):
    """
    Covariates effect.

    Parameters
    ----------
    *coefficients : float
        Coefficients of the covariates effect.
    """

    def __init__(self, coefficients=(None,)):
        super().__init__(**{f"coef_{i + 1}": v for i, v in enumerate(coefficients)})

    @property
    def nb_coef(self):
        """
        The number of coefficients

        Returns
        -------
        int
        """
        return self.nb_params

    def g(self, covar):
        """
        Compute the covariates effect.
        If covar.shape : () or (nb_coef,) => out.shape : (), float
        If covar.shape : (m, nb_coef) => out.shape : (m, 1)
        """
        arr_covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        if arr_covar.ndim > 2:
            raise ValueError(
                f"Invalid covar shape. Expected (nb_coef,) or (m, nb_coef) but got {arr_covar.shape}"
            )
        covar_nb_coef = arr_covar.size if arr_covar.ndim <= 1 else arr_covar.shape[-1]
        if covar_nb_coef != self.nb_coef:
            raise ValueError(
                f"Invalid covar. Number of covar does not match number of coefficients. Got {self.nb_coef} nb_coef but covar shape is {arr_covar.shape}"
            )
        g = np.exp(np.sum(self.params * arr_covar, axis=-1, keepdims=True))  # (m, 1)
        if arr_covar.ndim <= 1:
            return np.float64(g.item())
        return g

    def jac_g(self, covar, asarray=False):
        """
        Compute the Jacobian of the covariates effect.
        If covar.shape : () or (nb_coef,) => out.shape : (nb_coef,)
        If covar.shape : (m, nb_coef) => out.shape : (nb_coef, m, 1)
        """
        arr_covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        g = self.g(arr_covar)  # () or (m, 1)
        jac = arr_covar.T.reshape(self.nb_coef, -1, 1) * g  # (nb_coef, m, 1)
        if arr_covar.ndim <= 1:
            jac = jac.reshape(self.nb_coef)  # (nb_coef,) or (nb_coef, m, 1)
        if not asarray:
            return np.unstack(jac, axis=0)  # tuple
        return jac  # (nb_coef, m, 1)


class LifetimeRegression(FittableParametricLifetimeModel, ABC):
    """
    Base class for lifetime regression.
    """

    def __init__(self, baseline, coefficients=(None,)):
        super().__init__()
        self.covar_effect = _CovarEffect(coefficients)
        self.baseline = baseline

    @property
    def coefficients(self):
        """Coefficients of the regression.

        Returns
        -------
        ndarray
        """
        return self.covar_effect.params

    @property
    def nb_coef(self):
        """Number of coefficients.

        Returns
        -------
        int
        """
        return self.covar_effect.nb_params

    def sf(self, time, covar):
        """
        The survival function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return super().sf(time, covar)

    def isf(self, probability, covar):
        """
        The inverse survival function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given probability value(s).
        """
        cumulative_hazard_rate = -np.log(probability + 1e-6)  # avoid division by zero
        return self.ichf(cumulative_hazard_rate, covar)

    def cdf(self, time, covar):
        """
        The cumulative density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return super().cdf(time, covar)

    def pdf(self, time, covar):
        """
        The probility density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return super().pdf(time, covar)

    def ppf(self, probability, covar):
        """
        The percent point function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given probability value(s).
        """
        return super().ppf(probability,covar)

    def mrl(self, time, covar):
        """
        The mean residual life.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return super().mrl(time,covar)

    def ls_integrate(self, func, a, b, covar, deg: int = 10):
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
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.
        deg : int, default 10
            Degree of the polynomials interpolation

        Returns
        -------
        np.ndarray
            Lebesgue-Stieltjes integral of func from `a` to `b`.
        """
        return super().ls_integrate(func, a, b, covar, deg=deg)

    def moment(self, n, covar):
        """
        n-th order moment

        Parameters
        ----------
        n : int
            order of the moment, at least 1.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.

        Returns
        -------
        np.float64 or np.ndarray
        """
        return super().moment(n, covar)

    def mean(self, covar):
        """
        The mean.

        Parameters
        ----------
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.

        Returns
        -------
        np.float64 or np.ndarray
        """
        return super().mean(covar)

    def var(self, covar):
        """
        The variance.

        Parameters
        ----------
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.

        Returns
        -------
        np.float64 or np.ndarray
        """
        return super().var(covar)

    def median(self, covar):
        """
        The median.

        Parameters
        ----------
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.

        Returns
        -------
        np.float64 or np.ndarray
        """
        return super().median(covar)

    def jac_sf(
        self,
        time,
        covar,
        asarray=False,
    ):
        """
        The jacobian of the survival function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        jac = -self.jac_chf(time, covar, asarray=True) * self.sf(time, covar)
        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_cdf(self, time, covar, asarray=False):
        """
        The jacobian of the cumulative density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        jac = -self.jac_sf(time, covar, asarray=True)
        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_pdf(self, time, covar, asarray=False):
        """
        The jacobian of the probability density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        jac = self.jac_hf(time, covar, asarray=True) * self.sf(
            time, covar
        ) + self.jac_sf(time, covar, asarray=True) * self.hf(time, covar)
        if not asarray:
            return np.unstack(jac)
        return jac

    def rvs(
        self,
        size: int,
        covar,
        nb_assets=None,
        return_event=False,
        return_entry=False,
        seed=None,
    ):
        """
        Random variable sampling.

        Parameters
        ----------
        size : int
            Size of the generated sample.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)
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
            The sample values. If either ``return_event`` or ``random_entry`` is True, returns a tuple containing
            the time values followed by event values, entry values or both.
        """
        return super().rvs(
            size,
            covar,
            nb_assets=nb_assets,
            return_event=return_event,
            return_entry=return_entry,
            seed=seed,
        )

    def _get_initial_params(
        self, time, covar, event=None, entry=None
    ):
        self.covar_effect = _CovarEffect(
            (None,) * np.atleast_2d(np.asarray(covar)).shape[-1]
        )  # changes params structure depending on number of covar
        param0 = np.zeros_like(self.params, dtype=np.float64)
        param0[-self.baseline.params.size :] = self.baseline._get_initial_params(
            time, event=None, entry=None
        )  # recursion in case of PPH(AFT(...))
        return param0

    def _get_params_bounds(self):
        lb = np.concatenate(
            (
                np.full(self.covar_effect.nb_params, -np.inf),
                self.baseline._get_params_bounds().lb,  # baseline has _params_bounds according to typing
            )
        )
        ub = np.concatenate(
            (
                np.full(self.covar_effect.nb_params, np.inf),
                self.baseline._get_params_bounds().ub,
            )
        )
        return Bounds(lb, ub)

    def fit(
        self,
        time,
        covar,
        event=None,
        entry=None,
        optimizer_options=None,
    ):
        """
        Estimation of the regression parameters from lifetime data.

        Parameters
        ----------
        time : 1d array
            Observed lifetime values.
        covar : 1d or 2d array
            Covariates values. 1d array is valid if the regression has one coefficient.
            Otherwise it must be an 2d array of shape ``(m, nb_coef)``.
        event : 1d array of bool, default is None
            Boolean indicators tagging lifetime values as right censored or complete.
        entry : 1d array, default is None
            Left truncations applied to lifetime values.
        optimizer_options : dict, default is None
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
        """
        return super().fit(time, covar, event=event, entry=entry, optimizer_options=optimizer_options)

    def fit_from_interval_censored_lifetimes(
        self,
        time_inf,
        time_sup,
        covar,
        entry=None,
        optimizer_options=None,
    ):
        """
        Estimation of the regression parameters from interval censored lifetime data.

        Parameters
        ----------
        time_inf : 1d array
            Observed lifetime lower bounds.
        time_sup : 1d array
            Observed lifetime upper bounds.
        covar : 1d or 2d array
            Covariates values. 1d array is valid if the regression has one coefficient.
            Otherwise it must be an 2d array of shape ``(m, nb_coef)``.
        entry : 1d array, default is None
            Left truncations applied to lifetime values.
        optimizer_options : dict, default is None
            Extra arguments used by `scipy.minimize`. Default values are:
                - `method` : `"L-BFGS-B"`
                - `contraints` : `()`
                - `tol` : `None`
                - `callback` : `None`
                - `options` : `None`
                - `bounds` : `self.params_bounds`
                - `x0` : `self.init_params`

        Notes
        -----
        Where `time_inf == time_sup`, lifetimes are complete.

        Returns
        -------
        Self
            The current object with the estimated parameters setted inplace.
        """
        return super().fit_from_interval_censored_lifetimes(
            time_inf, time_sup, covar, entry=entry, optimizer_options=optimizer_options
        )

    def freeze(self, covar):
        """
        Freeze regression covar.

        Parameters
        ----------
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.

        Returns
        -------
        FrozenParametricModel
        """
        return FrozenParametricModel(self, covar)


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
    covar_effect : _CovarEffect
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

    def hf(self, time, covar):
        """
        The hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.covar_effect.g(covar) * self.baseline.hf(time)

    def chf(self, time, covar):
        """
        The cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.covar_effect.g(covar) * self.baseline.chf(time)

    def ichf(self, cumulative_hazard_rate, covar):
        """
        Inverse cumulative hazard function.

        Parameters
        ----------
        cumulative_hazard_rate : float or np.ndarray
            Cumulative hazard rate value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given cumulative hazard rate(s).
        """
        return self.baseline.ichf(cumulative_hazard_rate / self.covar_effect.g(covar))

    def dhf(self, time, covar):
        """
        The derivative of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.covar_effect.g(covar) * self.baseline.dhf(time)

    def jac_hf(self, time, covar, asarray=False):
        """
        The jacobian of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """

        time = np.asarray(time)  # (), (n,) or (m, n)
        covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        out_shape = _broadcast_time_covar_shapes(
            time.shape, covar.shape
        )  # (), (n,) or (m, n)
        time, covar = _broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True)  # (nb_coef, m, 1)

        baseline_hf = self.baseline.hf(time)  # (m, n)
        # p == baseline.nb_params
        baseline_jac_hf = self.baseline.jac_hf(time, asarray=True)  # (p, m, n)
        jac_g = np.repeat(
            jac_g, baseline_hf.shape[-1], axis=-1
        )  # (nb_coef, m, n) necessary to concatenate

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

    def jac_chf(self, time, covar, asarray=False):
        """
        The jacobian of the cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """

        time = np.asarray(time)  # (), (n,) or (m, n)
        covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        out_shape = _broadcast_time_covar_shapes(
            time.shape, covar.shape
        )  # (), (n,) or (m, n)
        time, covar = _broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True)  # (nb_coef, m, 1)
        baseline_chf = self.baseline.chf(time)  # (m, n)
        #  p == baseline.nb_params
        baseline_jac_chf = self.baseline.jac_chf(time, asarray=True)  # (p, m, n)
        jac_g = np.repeat(
            jac_g, baseline_chf.shape[-1], axis=-1
        )  # (nb_coef, m, n) necessary to concatenate

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
    covar_effect : _CovarEffect
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

    def hf(self, time, covar):
        """
        The hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.hf(t0) / self.covar_effect.g(covar)

    def chf(self, time, covar):
        """
        The cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.chf(t0)

    def ichf(self, cumulative_hazard_rate, covar):
        """
        Inverse cumulative hazard function.

        Parameters
        ----------
        cumulative_hazard_rate : float or np.ndarray
            Cumulative hazard rate value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given cumulative hazard rate(s).
        """
        return self.covar_effect.g(covar) * self.baseline.ichf(
            cumulative_hazard_rate
        )

    def dhf(self, time, covar):
        """
        The derivative of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.dhf(t0) / self.covar_effect.g(covar) ** 2

    def jac_hf(self, time, covar, asarray=False):
        """
        The jacobian of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """
        time = np.asarray(time)  # (), (n,) or (m, n)
        covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        out_shape = _broadcast_time_covar_shapes(
            time.shape, covar.shape
        )  # (), (n,) or (m, n)
        time, covar = _broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True)  # (nb_coef, m, 1)
        t0 = time / g  # (m, n)
        # p == baseline.nb_params
        baseline_jac_hf_t0 = self.baseline.jac_hf(t0, asarray=True)  # (p, m, n)
        baseline_hf_t0 = self.baseline.hf(t0)  # (m, n)
        baseline_dhf_t0 = self.baseline.dhf(t0)  # (m, n)
        jac_g = np.repeat(jac_g, baseline_hf_t0.shape[-1], axis=-1)  # (nb_coef, m, n)

        jac = np.concatenate(
            (
                -jac_g
                / g**2
                * (
                    baseline_hf_t0 + t0 * baseline_dhf_t0
                ),  # (nb_coef, m, n) necessary to concatenate
                baseline_jac_hf_t0 / g,  # (p, m, n)
            ),
            axis=0,
        )  # (p + nb_coef, m, n)

        jac = jac.reshape((self.nb_params,) + out_shape)
        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_chf(self, time, covar, asarray=False):
        """
        The jacobian of the cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n_values,)`` or ``(n_assets, n_values)``.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)
        asarray : bool, default is False

        Returns
        -------
        np.float64, np.ndarray or tuple of np.float64 or np.ndarray
            The derivatives with respect to each parameter. If ``asarray`` is False, the function returns a tuple containing
            the same number of elements as parameters. If ``asarray`` is True, the function returns an ndarray
            whose first dimension equals the number of parameters. This output is equivalent to applying ``np.stack`` on the output
            tuple when ``asarray`` is False.
        """

        time = np.asarray(time)  # (), (n,) or (m, n)
        covar = np.asarray(covar)  # (), (nb_coef,) or (m, nb_coef)
        out_shape = _broadcast_time_covar_shapes(
            time.shape, covar.shape
        )  # (), (n,) or (m, n)
        time, covar = _broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True)  # (nb_coef, m, 1)
        t0 = time / g  #  (m, n)
        # p == baseline.nb_params
        baseline_jac_chf_t0 = self.baseline.jac_chf(
            t0, asarray=True
        )  # (p, m, n)
        baseline_hf_t0 = self.baseline.hf(t0)  #  (m, n)
        jac_g = np.repeat(
            jac_g, baseline_hf_t0.shape[-1], axis=-1
        )  # (nb_coef, m, n) necessary to concatenate

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
