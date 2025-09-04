"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC

import numpy as np
from scipy.optimize import Bounds

from relife import ParametricModel

from ._base import FittableParametricLifetimeModel, FrozenParametricLifetimeModel


def broadcast_time_covar(time, covar):
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


def broadcast_time_covar_shapes(time_shape, covar_shape):
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


class CovarEffect(ParametricModel):
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
            raise ValueError(f"Invalid covar shape. Expected (nb_coef,) or (m, nb_coef) but got {arr_covar.shape}")
        covar_nb_coef = arr_covar.size if arr_covar.ndim <= 1 else arr_covar.shape[-1]
        if covar_nb_coef != self.nb_coef:
            raise ValueError(
                f"Invalid covar. Number of covar does not match number of coefficients. Got {self.nb_coef} nb_coef but covar shape is {arr_covar.shape}"
            )
        g = np.exp(np.sum(self.params * arr_covar, axis=-1, keepdims=True))  # (m, 1)
        if arr_covar.ndim <= 1:
            return np.float64(g.item())
        return g

    def jac_g(self, covar, *, asarray=False):
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
    Base class for regression model.

    At least one positional covar arg and 0 or more additional args (variable number)
    see : https://peps.python.org/pep-0646/#unpacking-unbounded-tuple-types*

    Note:
    LifetimeRegression does not preserve generic : at the moment, additional args are supposed to be always float | NDArray[np.float64]
    """

    def __init__(self, baseline, coefficients=(None,)):
        super().__init__()
        self.covar_effect = CovarEffect(coefficients)
        self.baseline = baseline
        self.fitting_results = None

    @property
    def coefficients(self):
        """Get the coefficients values of the covariate effect.

        Returns
        -------
        ndarray
        """
        return self.covar_effect.params

    @property
    def nb_coef(self):
        """The number of coefficients.

        Returns
        -------
        int
        """
        return self.covar_effect.nb_params

    def sf(self, time, covar, *args):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return super().sf(time, covar, *args)

    def isf(self, probability, covar, *args):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given probability value(s).
        """
        cumulative_hazard_rate = -np.log(probability + 1e-6)  # avoid division by zero
        return self.ichf(cumulative_hazard_rate, covar, *args)

    def cdf(self, time, covar, *args):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return super().cdf(time, *(covar, *args))

    def pdf(self, time, covar, *args):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return super().pdf(time, *(covar, *args))

    def ppf(self, probability, covar, *args):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given probability value(s).
        """
        return super().ppf(probability, *(covar, *args))

    def mrl(self, time, covar, *args):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return super().mrl(time, *(covar, *args))

    def ls_integrate(self, func, a, b, covar, *args, deg: int = 10):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.
        deg : int, default 10
            Degree of the polynomials interpolation

        Returns
        -------
        np.ndarray
            Lebesgue-Stieltjes integral of func from `a` to `b`.
        """
        return super().ls_integrate(func, a, b, *(covar, *args), deg=deg)

    def moment(self, n, covar, *args):
        """
        n-th order moment

        Parameters
        ----------
        n : int
            order of the moment, at least 1.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        return super().moment(n, *(covar, *args))

    def mean(self, covar, *args):
        """
        The mean.

        Parameters
        ----------
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        return super().mean(*(covar, *args))

    def var(self, covar, *args):
        """
        The variance.

        Parameters
        ----------
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        return super().var(*(covar, *args))

    def median(self, covar, *args):
        """
        The median.

        Parameters
        ----------
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
        """
        return super().median(*(covar, *args))

    def jac_sf(
        self,
        time,
        covar,
        *args,
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
        jac = -self.jac_chf(time, covar, *args, asarray=True) * self.sf(time, covar, *args)
        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_cdf(self, time, covar, *args, asarray=False):
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
        jac = -self.jac_sf(time, covar, *args, asarray=True)
        if not asarray:
            return np.unstack(jac)
        return jac

    def jac_pdf(self, time, covar, *args, asarray=False):
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
        jac = self.jac_hf(time, covar, *args, asarray=True) * self.sf(time, covar, *args) + self.jac_sf(
            time, covar, *args, asarray=True
        ) * self.hf(time, covar, *args)
        if not asarray:
            return np.unstack(jac)
        return jac

    def rvs(self, size: int, covar, *args, nb_assets=None, return_event=False, return_entry=False, seed=None):
        """
        Random variable sampling.

        Parameters
        ----------
        size : int
            Size of the generated sample.
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)
        *args : float or np.ndarray
            Additional arguments needed by the model.
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
            size, *(covar, *args), nb_assets=nb_assets, return_event=return_event, return_entry=return_entry, seed=seed
        )

    def _get_initial_params(self, time, covar, *args, event=None, entry=None, departure=None):
        self.covar_effect = CovarEffect(
            (None,) * covar.shape[-1]
        )  # changes params structure depending on number of covar
        param0 = np.zeros_like(self.params, dtype=np.float64)
        param0[-self.baseline.params.size :] = self.baseline._get_initial_params(
            time, *args, event=None, entry=None, departure=None
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
        *args,
        event=None,
        entry=None,
        departure=None,
        **options,
    ):
        """
        Estimation of the regression parameters from lifetime data.

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
        **options
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
        return super().fit(time, *(covar, *args), event=event, entry=entry, departure=departure, **options)

    def freeze(self, covar, *args):
        """
        Freeze regression covar and other arguments into the object data.

        Parameters
        ----------
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape ``(nb_coef,)`` or ``(m, nb_coef)``.
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        FrozenLifetimeRegression
        """
        return FrozenLifetimeRegression(self, covar, *args)


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

    def hf(self, time, covar, *args):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.covar_effect.g(covar) * self.baseline.hf(time, *args)

    def chf(self, time, covar, *args):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.covar_effect.g(covar) * self.baseline.chf(time, *args)

    def ichf(self, cumulative_hazard_rate, covar, *args):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given cumulative hazard rate(s).
        """
        return self.baseline.ichf(cumulative_hazard_rate / self.covar_effect.g(covar), *args)

    def dhf(self, time, covar, *args):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.covar_effect.g(covar) * self.baseline.dhf(time, *args)

    def jac_hf(self, time, covar, *args, asarray=False):
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

    def jac_chf(self, time, covar, *args, asarray=False):
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

    def hf(self, time, covar, *args):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.hf(t0, *args) / self.covar_effect.g(covar)

    def chf(self, time, covar, *args):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.chf(t0, *args)

    def ichf(self, cumulative_hazard_rate, covar, *args):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given cumulative hazard rate(s).
        """
        return self.covar_effect.g(covar) * self.baseline.ichf(cumulative_hazard_rate, *args)

    def dhf(self, time, covar, *args):
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
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.dhf(t0, *args) / self.covar_effect.g(covar) ** 2

    def jac_hf(self, time, covar, *args, asarray=False):
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

    def jac_chf(self, time, covar, *args, asarray=False):
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


class FrozenLifetimeRegression(FrozenParametricLifetimeModel):
    r"""
    Frozen lifetime regression.

    Parameters
    ----------
    regression : LifetimeRegression
        Any lifetime regression.
    covar : float or np.ndarray
        Covariate values to be frozen.
    *args : float or np.ndarray
        Additional arguments needed by the model to be frozen.

    Attributes
    ----------
    unfrozen_model : LifetimeRegression
        The unfrozen regression model.
    args : tuple of float or np.ndarray
        All the frozen arguments given and necessary to compute model functions.
    nb_assets : int
        Number of assets passed in frozen arguments. The data is mainly used to control numpy broadcasting and may not
        interest an user.

    Warnings
    --------
    This class is documented for the purpose of clarity and mainly address contributors or advance users. Actually, the
    recommanded way to instanciate a ``FrozenLifetimeRegression`` is use to ``freeze`` factory function.

    """

    def __init__(self, regression, covar, *args):
        super().__init__(regression, *(covar, *args))

    def unfreeze(self) -> LifetimeRegression:
        return self.unfrozen_model

    @property
    def nb_coef(self):
        """
        The number of coefficients

        Returns
        -------
        int
        """
        return self.unfrozen_model.nb_coef

    @property
    def covar(self):
        return self.args[0]

    @covar.setter
    def covar(self, value):
        self.args = (value,) + self.args[1:]

    def dhf(self, time):
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
        return self.unfrozen_model.dhf(time, self.args[0], *self.args[1:])

    def jac_hf(self, time, asarray=False):
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
        return self.unfrozen_model.jac_hf(time, self.args[0], *self.args[1:], asarray=asarray)

    def jac_chf(self, time, asarray=False):
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
        return self.unfrozen_model.jac_chf(time, self.args[0], *self.args[1:], asarray=asarray)

    def jac_sf(self, time, asarray=False):
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
        return self.unfrozen_model.jac_sf(time, self.args[0], *self.args[1:], asarray=asarray)

    def jac_cdf(self, time, asarray=False):
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
        return self.unfrozen_model.jac_cdf(time, self.args[0], *self.args[1:], asarray=asarray)

    def jac_pdf(self, time, asarray: bool = False):
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
        return self.unfrozen_model.jac_pdf(time, self.args[0], *self.args[1:], asarray=asarray)
