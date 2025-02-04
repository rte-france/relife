"""
This module defines probability functions used in regression

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from typing import Optional, TypeVarTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds
from typing_extensions import override

from relife.data import LifetimeData
from relife.model import ParametricLifetimeModel, ParametricModel
from relife.typing import ModelArgs

Ts = TypeVarTuple("Ts")


class CovarEffect(ParametricModel):
    """
    Covariate effect.

    Parameters
    ----------
    coef : tuple of float or tuple of None, optional
        Coefficients used to parametrized the covariate effect.
        If None is provided, the coefficients values will be set to np.nan.

    """

    def __init__(self, coef: tuple[float, ...] | tuple[None] = (None,)):
        super().__init__()
        self.new_params(**{f"coef_{i}": v for i, v in enumerate(coef)})

    def g(self, covar: NDArray[np.float64]) -> NDArray[np.float64]:
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

    def jac_g(self, covar: NDArray[np.float64]) -> NDArray[np.float64]:
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


class Regression(
    ParametricLifetimeModel[NDArray[np.float64], *ModelArgs],
    ABC,
):
    """
    Base class for regression models.

    Parameters
    ----------
    baseline : ParametricLifetimeModel
        Any parametric lifetime model to serve as the baseline.
    coef : tuple of floats (values can be None), optional
        Coefficients values of the covariate effects.

    See Also
    --------
    regression.AFT : AFT regression
    """

    def __init__(
        self,
        baseline: ParametricLifetimeModel[*ModelArgs],
        coef: tuple[float, ...] | tuple[None] = (None,),
    ):
        super().__init__()
        self.compose_with(
            covar_effect=CovarEffect(coef),
            baseline=baseline,
        )

    def init_params(
        self,
        lifetime_data: LifetimeData,
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> None:
        """
        Initialize parameters for the regression model.

        Parameters
        ----------
        lifetime_data : LifetimeData
            The lifetime data used to initialize the baseline model.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values. Should have shape (k, ) or (m, k) where m is the number of assets and k is the number of covariates.
        *args : variable number of arguments
            Any additional arguments needed by the baseline model.
        """
        self.covar_effect.new_params(
            **{f"coef_{i}": 0.0 for i in range(covar.shape[-1])}
        )
        self.baseline.init_params(lifetime_data, *args)

    @property
    def params_bounds(self) -> Bounds:
        """
        Bounds of the parameters.

        Returns
        -------
        Bounds
            The lower and upper bounds for the parameters.
        """
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

    @property
    def _default_hess_scheme(self) -> str:
        return self.baseline._default_hess_scheme

    @override
    def sf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """
        Survival function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Survival probabilities at each given time.
        """
        return super().sf(time, covar, *args)

    @override
    def isf(
        self,
        probability: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """
        Inverse survival function.

        Parameters
        ----------
        probability : float or ndarray, shape (n, ) or (m, n)
            Survival probabilities.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Time values corresponding to the given survival probabilities.
        """
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, covar, *args)

    @override
    def cdf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """
        Cumulative distribution function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Cumulative probabilities at each given time.
        """
        return super().cdf(time, covar, *args)

    def pdf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """
        Probability density function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Probability densities at each given time.
        """
        return super().pdf(time, covar, *args)

    @override
    def ppf(
        self,
        probability: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """
        Percent point function (inverse of cdf).

        Parameters
        ----------
        probability : float or ndarray, shape (n, ) or (m, n)
            Cumulative probabilities.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Time values corresponding to the given cumulative probabilities.
        """
        return super().ppf(probability, covar, *args)

    @override
    def mrl(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """
        Mean residual life.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Mean residual life values at each given time.
        """
        return super().mrl(time, covar, *args)

    @override
    def rvs(
        self,
        covar: NDArray[np.float64],
        *args: *ModelArgs,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ):
        """
        Random variates.

        Parameters
        ----------
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.
        size : int, optional
            Number of random variates to generate.
        seed : int, optional
            Seed for random number generator.

        Returns
        -------
        np.ndarray
            Random variates.
        """
        return super().rvs(covar, *args, size=size, seed=seed)

    @override
    def mean(
        self, covar: NDArray[np.float64], *args: *ModelArgs
    ) -> NDArray[np.float64]:
        """
        Mean of the distribution.

        Parameters
        ----------
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        np.ndarray
            Mean values.
        """
        return super().mean(covar, *args)

    @override
    def var(self, covar: NDArray[np.float64], *args: *ModelArgs) -> NDArray[np.float64]:
        """
        Variance of the distribution.

        Parameters
        ----------
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        np.ndarray
            Variance values.
        """
        return super().var(covar, *args)

    @override
    def median(
        self, covar: NDArray[np.float64], *args: *ModelArgs
    ) -> NDArray[np.float64]:
        """
        Median of the distribution.

        Parameters
        ----------
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        np.ndarray
            Median values.
        """
        return super().median(covar, *args)

    @abstractmethod
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """
        Jacobian of the hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        np.ndarray of shape (n, nb_params) or (m, n, nb_params)
            The values of the Jacobian at each n points (eventually for m assets).
        """

    @abstractmethod
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """
        Jacobian of the cumulative hazard function.

        Parameters
        ----------
        time : np.ndarray of shape (n, ) or (m, n)
            Elapsed time.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        np.ndarray of shape (n, nb_params) or (m, n, nb_params)
            The values of the Jacobian at each n points (eventually for m assets).
        """

    @abstractmethod
    def dhf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """
        Derivative of the hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Derivative values with respect to time.
        """

    def jac_sf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """Jacobian of the survival function

        Jacobian with respect to model parameters.

        Parameters
        ----------
        time : np.darray of shape (n, ) or (m, n)
            Elapsed time.
        covar : np.darray of shape (k, ) or (m, k)
            Covariates values.
        *args : variable number of np.darray
            Any other variables needed by the model

        Returns
        -------
        np.ndarray of shape (n, nb_params) or (m, n, nb_params)
            The values of the jacobian at each n points (eventually for m assets).

        """
        return -self.jac_chf(time, covar, *args) * self.sf(time, covar, *args)

    def jac_cdf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """Jacobian of the cumulative distribution function.

        Jacobian with respect to model parameters.

        Parameters
        ----------
        time : np.darray of shape (n, ) or (m, n)
            Elapsed time.
        covar : np.darray of shape (k, ) or (m, k)
            Covariates values.
        *args : variable number of np.darray
            Any other variables needed by the model

        Returns
        -------
        np.ndarray of shape (n, nb_params) or (m, n, nb_params)
            The values of the jacobian at each n points (eventually for m assets).
        """
        return -self.jac_sf(time, covar, *args)

    def jac_pdf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """Jacobian of the probability density function.

        Jacobian with respect to model parameters.

        Parameters
        ----------
        time : np.darray of shape (n, ) or (m, n)
            Elapsed time.
        covar : np.darray of shape (k, ) or (m, k)
            Covariates values.
        *args : variable number of np.darray
            Any other variables needed by the model

        Returns
        -------
        np.ndarray of shape (n, nb_params) or (m, n, nb_params)
            The values of the jacobian at each n points (eventually for m assets).
        """

        return self.jac_hf(time, covar, *args) * self.sf(
            time, covar, *args
        ) + self.jac_sf(time, covar, *args) * self.hf(time, covar, *args)


class ProportionalHazard(Regression):
    r"""
    Proportional Hazard regression model.

    The cumulative hazard function :math:`H` is linked to the multiplier
    function :math:`g` by the relation:

    .. math::

        H(t, x) = g(\beta, x) H_0(t) = e^{\beta \cdot x} H_0(t)

    where :math:`x` is a vector of covariates, :math:`\beta` is the coefficient
    vector of the effect of covariates, :math:`H_0` is the baseline cumulative
    hazard function.

    Parameters
    ----------
    baseline : ParametricLifetimeModel
        Any parametric lifetime model to serve as the baseline.
    coef : tuple of floats (values can be None), optional
        Coefficients values of the covariate effects.

    References
    ----------
    .. [1] Sun, J. (2006). The statistical analysis of interval-censored failure
        time data (Vol. 3, No. 1). New York: springer.

    See Also
    --------
    regression.AFT : AFT regression
    """

    def __init__(
        self,
        baseline: ParametricLifetimeModel[*ModelArgs],
        coef: tuple[float, ...] | tuple[None] = (None,),
    ):
        super().__init__(baseline, coef)

    def hf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """Hazard function.

        The hazard function of the regression

        Parameters
        ----------
        time : np.darray
            Elapsed time.
        covar : np.darray
            Covariates values.
        *args : variable number of np.darray
            Any other variables needed by the model


        Returns
        -------
        numpy array of floats
            Hazard values at each given time.

        Notes
        -----
        `time`, `covar` and any `*args` arrays must be broadcastable
        """
        return self.covar_effect.g(covar) * self.baseline.hf(time, *args)

    def chf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """Cumulative hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Cumulative hazard values at each given time.
        """
        return self.covar_effect.g(covar) * self.baseline.chf(time, *args)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """Inverse cumulative hazard function.

        Parameters
        ----------
        cumulative_hazard_rate : float or ndarray, shape (n, ) or (m, n)
            Cumulative hazard rate values.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Inverse cumulative hazard values, i.e., time.
        """
        return self.baseline.ichf(
            cumulative_hazard_rate / self.covar_effect.g(covar), *args
        )

    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        return np.column_stack(
            (
                self.covar_effect.jac_g(covar) * self.baseline.hf(time, *args),
                self.covar_effect.g(covar) * self.baseline.jac_hf(time, *args),
            )
        )

    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        return np.column_stack(
            (
                self.covar_effect.jac_g(covar) * self.baseline.chf(time, *args),
                self.covar_effect.g(covar) * self.baseline.jac_chf(time, *args),
            )
        )

    def dhf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        return self.covar_effect.g(covar) * self.baseline.dhf(time, *args)


class AFT(Regression):
    r"""
    Accelerated failure time regression.

    The cumulative hazard function :math:`H` is linked to the multiplier
    function :math:`g` by the relation:

    .. math::

        H(t, x) = H_0\left(\dfrac{t}{g(\beta, x)}\right) = H_0(t e^{- \beta
        \cdot x})

    where :math:`x` is a vector of covariates, :math:`\beta` is the coefficient
    vector of the effect of covariates, :math:`H_0` is the baseline cumulative
    hazard function.

    Parameters
    ----------
    baseline : ParametricLifetimeModel
        Any parametric lifetime model to serve as the baseline.
    coef : tuple of floats (values can be None), optional
        Coefficients values of the covariate effects.

    References
    ----------
    .. [1] Kalbfleisch, J. D., & Prentice, R. L. (2011). The statistical
        analysis of failure time data. John Wiley & Sons.
    """

    def __init__(
        self,
        baseline: ParametricLifetimeModel[*ModelArgs],
        coef: tuple[float, ...] | tuple[None] = (None,),
    ):
        super().__init__(baseline, coef)

    def hf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """Hazard function.

        The hazard function of the regression.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Hazard values at each given time.

        Notes
        -----
        `time`, `covar`, and any `*args` arrays must be broadcastable.
        """
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.hf(t0, *args) / self.covar_effect.g(covar)

    def chf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """Cumulative hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Cumulative hazard values at each given time.
        """
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.chf(t0, *args)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """Inverse cumulative hazard function.

        Parameters
        ----------
        cumulative_hazard_rate : float or ndarray, shape (n, ) or (m, n)
            Cumulative hazard rate values.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Inverse cumulative hazard values, i.e., time.
        """
        return self.covar_effect.g(covar) * self.baseline.ichf(
            cumulative_hazard_rate, *args
        )

    def jac_hf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """Jacobian of the hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        np.ndarray of shape (n, nb_params) or (m, n, nb_params)
            The values of the Jacobian at each n points (eventually for m assets).
        """
        t0 = time / self.covar_effect.g(covar)
        return np.column_stack(
            (
                -self.covar_effect.jac_g(covar)
                / self.covar_effect.g(covar) ** 2
                * (self.baseline.hf(t0, *args) + t0 * self.baseline.dhf(t0, *args)),
                self.baseline.jac_hf(t0, *args) / self.covar_effect.g(covar),
            )
        )

    def jac_chf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """Jacobian of the cumulative hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        np.ndarray of shape (n, nb_params) or (m, n, nb_params)
            The values of the Jacobian at each n points (eventually for m assets).
        """
        t0 = time / self.covar_effect.g(covar)
        return np.column_stack(
            (
                -self.covar_effect.jac_g(covar)
                / self.covar_effect.g(covar)
                * t0
                * self.baseline.hf(t0, *args),
                self.baseline.jac_chf(t0, *args),
            )
        )

    def dhf(
        self,
        time: float | NDArray[np.float64],
        covar: NDArray[np.float64],
        *args: *ModelArgs,
    ) -> NDArray[np.float64]:
        """Derivative of the hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time.
        covar : np.ndarray of shape (k, ) or (m, k)
            Covariate values.
        *args : variable number of np.ndarray
            Any other variables needed by the model.

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Derivative values with respect to time.
        """
        t0 = time / self.covar_effect.g(covar)
        return self.baseline.dhf(t0, *args) / self.covar_effect.g(covar) ** 2
