"""Lifetime regression

Notes
-----
This module contains two parametric lifetime regressions.
ProportionalHazard is not Cox regression (Cox is semiparametric).
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Literal, Self, final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds
from typing_extensions import overload, override

from relife.base import ParametricModel
from relife.typing import (
    AnyFloat,
    NumpyBool,
    NumpyFloat,
    ScipyMinimizeOptions,
    Seed,
)

from ._base import FittableParametricLifetimeModel
from ._distribution import LifetimeDistribution
from ._frozen import FrozenParametricLifetimeModel

__all__: list[str] = ["AcceleratedFailureTime", "ProportionalHazard"]


def _broadcast_time_covar(time: AnyFloat, covar: AnyFloat) -> tuple[NumpyFloat, NumpyFloat]:
    time = np.atleast_2d(np.asarray(time))  #  (m, n)
    covar = np.atleast_2d(np.asarray(covar))  #  (m, nb_coef)
    match (time.shape[0], covar.shape[0]):
        case (1, _):
            time = np.repeat(time, covar.shape[0], axis=0)
        case (_, 1):
            covar = np.repeat(covar, time.shape[0], axis=0)
        case (m1, m2) if m1 != m2:
            raise ValueError(f"Incompatible time and covar. time has {m1} nb_assets but covar has {m2} nb_assets")
        case _:
            pass
    return time, covar


def _broadcast_time_covar_shapes(time_shape: tuple[int, ...], covar_shape: tuple[int, ...]) -> tuple[int, ...]:
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
                raise ValueError(f"Invalid time and covar : time got {mt} nb assets but covar got {mc} nb assets")
            return max(mt, mc), n
        case [(mt, n), (mc, _)] if mt == mc:
            return mt, n
        case _:
            raise ValueError(f"Invalid time or covar shape. Got {time_shape} and {covar_shape}")


@final
class CovarEffect(ParametricModel):
    """
    Covariates effect.

    Parameters
    ----------
    *coefficients : float
        Coefficients of the covariates effect.
    """

    def __init__(self, coefficients: tuple[float | None, ...] = (None,)):
        super().__init__(**{f"coef_{i + 1}": v for i, v in enumerate(coefficients)})

    @property
    def nb_coef(self) -> int:
        """
        The number of coefficients

        Returns
        -------
        int
        """
        return self.nb_params

    def g(self, covar: AnyFloat) -> NumpyFloat:
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

    @overload
    def jac_g(self, covar: AnyFloat, asarray: Literal[True]) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_g(self, covar: AnyFloat, asarray: Literal[False]) -> NumpyFloat: ...
    @overload
    def jac_g(self, covar: AnyFloat, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    def jac_g(self, covar: AnyFloat, asarray: bool = False) -> tuple[NumpyFloat, ...] | NumpyFloat:
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


class LifetimeRegression(FittableParametricLifetimeModel[AnyFloat], ABC):
    """
    Base class for lifetime regression.
    """

    baseline: LifetimeDistribution
    covar_effect: CovarEffect

    def __init__(self, baseline: LifetimeDistribution, coefficients: tuple[float | None, ...] = (None,)):
        super().__init__()
        self.covar_effect = CovarEffect(coefficients)
        self.baseline = baseline

    @property
    def coefficients(self) -> NDArray[np.float64]:
        """Coefficients of the regression.

        Returns
        -------
        ndarray
        """
        return self.covar_effect.params

    @property
    def nb_coef(self) -> int:
        """Number of coefficients.

        Returns
        -------
        int
        """
        return self.covar_effect.nb_params

    @override
    def sf(self, time: AnyFloat, covar: AnyFloat) -> NumpyFloat:
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

    @override
    def isf(self, probability: AnyFloat, covar: AnyFloat) -> NumpyFloat:
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

    @override
    def cdf(self, time: AnyFloat, covar: AnyFloat) -> NumpyFloat:
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

    @override
    def pdf(self, time: AnyFloat, covar: AnyFloat) -> NumpyFloat:
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

    @override
    def ppf(self, probability: AnyFloat, covar: AnyFloat) -> NumpyFloat:
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
        return super().ppf(probability, covar)

    @override
    def mrl(self, time: AnyFloat, covar: AnyFloat) -> NumpyFloat:
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
        return super().mrl(time, covar)

    @override
    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: AnyFloat,
        b: AnyFloat,
        covar: AnyFloat,
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

    @override
    def moment(self, n: int, covar: AnyFloat) -> NumpyFloat:
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

    @override
    def mean(self, covar: AnyFloat) -> NumpyFloat:
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

    @override
    def var(self, covar: AnyFloat) -> NumpyFloat:
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

    @override
    def median(self, covar: AnyFloat) -> NumpyFloat:
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

    @overload
    def jac_sf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_sf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_sf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_sf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        asarray: bool = True,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat:
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

    @overload
    def jac_cdf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_cdf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_cdf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    def jac_cdf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        asarray: bool = True,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat:
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

    @overload
    def jac_pdf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_pdf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_pdf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_pdf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        asarray: bool = True,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat:
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
        jac = self.jac_hf(time, covar, asarray=True) * self.sf(time, covar) + self.jac_sf(
            time, covar, asarray=True
        ) * self.hf(time, covar)
        if not asarray:
            return np.unstack(jac)
        return jac

    @overload
    def rvs(
        self,
        size: int,
        covar: AnyFloat,
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
        covar: AnyFloat,
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
        covar: AnyFloat,
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
        covar: AnyFloat,
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
        covar: AnyFloat,
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
        covar: AnyFloat,
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

    def freeze(self, covar: AnyFloat) -> FrozenParametricLifetimeModel[AnyFloat]:
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
        return FrozenParametricLifetimeModel(self, covar)

    @property
    @override
    def params_bounds(self) -> Bounds:
        lb = np.concatenate(
            (
                np.full(self.covar_effect.nb_params, -np.inf),
                self.baseline.params_bounds.lb,  # baseline has _params_bounds according to typing
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
    def get_initial_params(
        self,
        time: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
    ) -> NDArray[np.float64]:
        if model_args is None:
            raise ValueError
        covar = model_args[0]
        self.covar_effect = CovarEffect(
            (None,) * np.atleast_2d(np.asarray(covar, dtype=np.float64)).shape[-1]
        )  # changes params structure depending on number of covar
        param0 = np.zeros_like(self.params, dtype=np.float64)
        param0[-self.baseline.params.size :] = self.baseline.get_initial_params(time)
        return param0

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
            raise ValueError("LifetimeRegression expects covar but model_args is None")
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
            raise ValueError("LifetimeRegression expects covar but model_args is None")
        covar = model_args[0]
        self.covar_effect = CovarEffect(
            (None,) * np.atleast_2d(np.asarray(covar, dtype=np.float64)).shape[-1]
        )  # changes params structure depending on number of covar
        return super().fit_from_interval_censored_lifetimes(
            time_inf, time_sup, model_args=model_args, entry=entry, optimizer_options=optimizer_options
        )


@final
class ProportionalHazard(LifetimeRegression):
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

    @override
    def hf(self, time: AnyFloat, covar: AnyFloat) -> NumpyFloat:
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

    @override
    def chf(self, time: AnyFloat, covar: AnyFloat) -> NumpyFloat:
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

    @override
    def ichf(self, cumulative_hazard_rate: AnyFloat, covar: AnyFloat) -> NumpyFloat:
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

    @override
    def dhf(self, time: AnyFloat, covar: AnyFloat) -> NumpyFloat:
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

    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_hf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        asarray: bool = True,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat:
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
        out_shape = _broadcast_time_covar_shapes(time.shape, covar.shape)  # (), (n,) or (m, n)
        time, covar = _broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True)  # (nb_coef, m, 1)

        baseline_hf = self.baseline.hf(time)  # (m, n)
        # p == baseline.nb_params
        baseline_jac_hf = self.baseline.jac_hf(time, asarray=True)  # (p, m, n)
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
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_chf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        asarray: bool = True,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat:
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
        out_shape = _broadcast_time_covar_shapes(time.shape, covar.shape)  # (), (n,) or (m, n)
        time, covar = _broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True)  # (nb_coef, m, 1)
        baseline_chf = self.baseline.chf(time)  # (m, n)
        #  p == baseline.nb_params
        baseline_jac_chf = self.baseline.jac_chf(time, asarray=True)  # (p, m, n)
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


@final
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

    @override
    def hf(self, time: AnyFloat, covar: AnyFloat) -> NumpyFloat:
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

    @override
    def chf(self, time: AnyFloat, covar: AnyFloat) -> NumpyFloat:
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

    @override
    def ichf(self, cumulative_hazard_rate: AnyFloat, covar: AnyFloat) -> NumpyFloat:
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
        return self.covar_effect.g(covar) * self.baseline.ichf(cumulative_hazard_rate)

    @override
    def dhf(self, time: AnyFloat, covar: AnyFloat) -> NumpyFloat:
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

    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_hf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_hf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        asarray: bool = True,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat:
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
        out_shape = _broadcast_time_covar_shapes(time.shape, covar.shape)  # (), (n,) or (m, n)
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
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: Literal[False],
    ) -> tuple[NumpyFloat, ...]: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: Literal[True],
    ) -> NumpyFloat: ...
    @overload
    def jac_chf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        *,
        asarray: bool,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat: ...
    @override
    def jac_chf(
        self,
        time: AnyFloat,
        covar: AnyFloat,
        asarray: bool = True,
    ) -> tuple[NumpyFloat, ...] | NumpyFloat:
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
        out_shape = _broadcast_time_covar_shapes(time.shape, covar.shape)  # (), (n,) or (m, n)
        time, covar = _broadcast_time_covar(time, covar)  # (m, n) and (m, nb_coef)

        g = self.covar_effect.g(covar)  # (m, 1)
        jac_g = self.covar_effect.jac_g(covar, asarray=True)  # (nb_coef, m, 1)
        t0 = time / g  #  (m, n)
        # p == baseline.nb_params
        baseline_jac_chf_t0 = self.baseline.jac_chf(t0, asarray=True)  # (p, m, n)
        baseline_hf_t0 = self.baseline.hf(t0)  #  (m, n)
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
