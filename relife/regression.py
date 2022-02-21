"""Survival regression models."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.optimize import Bounds, OptimizeResult

from .data import LifetimeData
from .parametric import ParametricLifetimeModel, FittingResult
from .distribution import ParametricLifetimeDistribution


def concordance_index(
    time: np.ndarray,
    event: np.ndarray,
    score: np.ndarray,
    include_tied_score: bool = True,
) -> Tuple[float, int, int, int]:
    """Computes the concordance index (c-index).

    The concordance index is a metric for evaluating the predictions of a
    survival regression model. It is the proportion of concordant
    pairs divided by the total number of possible evaluation pairs.

    Parameters
    ----------
    time : 1D array
        Array of time-to-event or durations.
    event : 1D array
        Array of event types coded as follows:

        - 0 if observation ends before the event has occurred (right censoring)
        - 1 if the event has occured

    score : 1D array
        Vector of individual risk scores (e.g. the median).

    include_tied_score : bool, optional
        Specifies whether ties in risk score are included in calculations, by
        default True.

    Returns
    -------
    Tuple[float, int, int, int]

        - cindex     : The computed concordance index (float)
        - concordant : Number of concordant pairs (int)
        - discordant : Number of discordant pairs (int)
        - tied_score : Number of pairs having tied estimated risks (int)
    """
    admissible = 0
    concordant = 0
    tied_score = 0

    event_observed = event == 1
    event_not_observed = ~event_observed

    t1 = time[event_observed]
    s1 = score[event_observed]

    t0 = time[event_not_observed]
    s0 = score[event_not_observed]

    for ti, si in zip(t1, s1):
        ind1 = ti < t1
        ind0 = ti <= t0
        admissible += np.sum(ind1) + np.sum(ind0)
        concordant += np.sum(si < s1[ind1]) + np.sum(si < s0[ind0])
        tied_score += np.sum(si == s1[ind1]) + np.sum(si == s0[ind0])

    if include_tied_score:
        cindex = (concordant + 0.5 * tied_score) / admissible
    else:
        cindex = concordant / (admissible - tied_score)

    discordant = admissible - concordant - tied_score

    return cindex, concordant, discordant, tied_score


@dataclass
class Regression(ParametricLifetimeModel):
    r"""Generic regression for parametric lifetime model.

    A generic class for survival regression model. The model includes a baseline
    lifetime model and a multiplier function :math:`g` that specifies the effect
    of the covariates (explanatory variables) on the baseline model:

    .. math::

        g(\beta, x) = e^{\beta \cdot x}
    """

    baseline: ParametricLifetimeModel  #: Baseline parametric lifetime model for the regression.
    beta: np.ndarray = None  #: Coefficients for the covariates.

    def __post_init__(self):
        if self.beta is not None:
            self.beta = np.asanyarray(self.beta, float)
            self.n_covar = self.beta.size

    @property
    def params(self) -> np.ndarray:
        return np.concatenate((self.beta, self.baseline.params))

    @property
    def n_params(self) -> int:
        return self.n_covar + self.baseline.n_params

    @property
    def _param_bounds(self) -> Bounds:
        lb = np.concatenate(
            (np.full(self.n_covar, -np.inf), self.baseline._param_bounds.lb)
        )
        ub = np.concatenate(
            (np.full(self.n_covar, np.inf), self.baseline._param_bounds.ub)
        )
        return Bounds(lb, ub)

    @property
    def _default_hess_scheme(self) -> str:
        return self.baseline._default_hess_scheme

    def _init_params(self, data: LifetimeData) -> np.ndarray:
        args = data.args
        if not len(data.args) > 0:
            raise ValueError(
                "`covar` argument is missing for regression model {}".format(
                    self.__class__.__name__
                )
            )
        if data.args[0].ndim != 2:
            raise ValueError("`covar` argument must be a 2d-array")
        covar, *args = args
        self.n_covar = covar.shape[1]
        data = LifetimeData(*data.astuple()[:-1], args)
        return np.concatenate(
            (np.zeros(self.n_covar), self.baseline._init_params(data))
        )

    def _dispatch(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatch the params array into two arrays.

        Parameters
        ----------
        params : 1D array
            Parameters array for the regression model.

        Returns
        -------
        Tuple[1D array, 1D array]
            The first array contains the beta parameters for the covariates, and the
            second the parameters of the baseline model.
        """
        return params[: self.n_covar], params[self.n_covar :]

    def _set_params(self, params: np.ndarray) -> None:
        self.beta, params0 = self._dispatch(params)
        self.baseline._set_params(params0)

    @classmethod
    def _g(cls, beta: np.ndarray, covar: np.ndarray) -> np.ndarray:
        """The multiplier that specifies the effect of covariates.

        Parameters
        ----------
        beta : 1D array
            The beta parameters of the covariates.
        covar : 2D array
            The covariates array.

        Returns
        -------
        float or 2D array
            The multiplier function evaluated at `beta` and `covar`.
        """
        return np.exp(np.sum(beta * covar, axis=1, keepdims=True))

    @classmethod
    def _jac_g(cls, beta: np.ndarray, covar: np.ndarray) -> np.ndarray:
        """Jacobian of the multiplier function.

        Parameters
        ----------
        beta : 1D array
            The `beta` parameters of the covariates.
        covar : 2D array
            The covariates array.

        Returns
        -------
        float or 2D array
            The jacobian of the multiplier function `g` evaluated at `beta` and `covar`.
        """
        return covar * cls._g(beta, covar)

    def fit(
        self,
        time: np.ndarray,
        event: np.ndarray = None,
        entry: np.ndarray = None,
        covar: np.ndarray = None,
        args: np.ndarray = (),
        params0: np.ndarray = None,
        method: str = None,
        **kwargs,
    ) -> Regression:
        """Fit the parametric survival regression model to lifetime data.

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
        covar : 2D array
            Array of covariates.
        args : float or 2D array, optional
            Extra arguments required by the baseline parametric lifetime model.
        params0 : 1D array, optional
            Initial guess, by default None.
        method : str, optional
            Type of solver (see scipy.optimize.minimize documentation), by
            default None.

        Returns
        -------
        self
            Return the fitted regression as the current object.
        """
        args = (covar, *args) if covar is not None else args
        data = LifetimeData(time, event, entry, args)
        self._fit(data, params0, method=method, **kwargs)
        return self

    def _set_fitting_result(
        self, opt: OptimizeResult, jac: np.ndarray, var: np.ndarray, data: LifetimeData
    ) -> None:
        cindex = concordance_index(data.time, data.event, self.median(*data.args))[0]
        self.result = RegressionFittingResult(opt, jac, var, data.time.size, cindex)


@dataclass
class RegressionFittingResult(FittingResult):
    """Class for the result of the fitted regression, inheriting from
    FittingResult.

    Used as the type for the instance attribute 'result' for an object of the
    Regression class, after fitting.
    """

    cindex: float


# Regression models


class AFT(Regression):
    r"""Accelerated Failure Time Regression Model.

    The cumulative hazard function :math:`H` is linked to the multiplier
    function :math:`g` by the relation:

    .. math::

        H(t, x) = H_0\left(\dfrac{t}{g(\beta, x)}\right) = H_0(t e^{- \beta
        \cdot x})

    where :math:`x` is a vector of covariates, :math:`\beta` is the coefficient
    vector of the effect of covariates, :math:`H_0` is the baseline cumulative
    hazard function.

    References
    ----------
    .. [1] Kalbfleisch, J. D., & Prentice, R. L. (2011). The statistical
        analysis of failure time data. John Wiley & Sons.
    """

    def _chf(
        self, params: np.ndarray, t: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        beta, params0 = self._dispatch(params)
        t0 = t / self._g(beta, covar)
        return self.baseline._chf(params0, t0, *args)

    def _hf(
        self, params: np.ndarray, t: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        beta, params0 = self._dispatch(params)
        t0 = t / self._g(beta, covar)
        return self.baseline._hf(params0, t0, *args) / self._g(beta, covar)

    def _dhf(
        self, params: np.ndarray, t: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        beta, params0 = self._dispatch(params)
        t0 = t / self._g(beta, covar)
        return self.baseline._dhf(params0, t0, *args) / self._g(beta, covar) ** 2

    def _jac_chf(
        self, params: np.ndarray, t: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        beta, params0 = self._dispatch(params)
        t0 = t / self._g(beta, covar)
        return np.column_stack(
            (
                -self._jac_g(beta, covar)
                / self._g(beta, covar)
                * t0
                * self.baseline._hf(params0, t0, *args),
                self.baseline._jac_chf(params0, t0, *args),
            )
        )

    def _jac_hf(
        self, params: np.ndarray, t: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        beta, params0 = self._dispatch(params)
        t0 = t / self._g(beta, covar)
        return np.column_stack(
            (
                -self._jac_g(beta, covar)
                / self._g(beta, covar) ** 2
                * (
                    self.baseline._hf(params0, t0, *args)
                    + t0 * self.baseline._dhf(params0, t0, *args)
                ),
                self.baseline._jac_hf(params0, t0, *args) / self._g(beta, covar),
            )
        )

    def _ichf(
        self, params: np.ndarray, v: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        beta, params0 = self._dispatch(params)
        return self._g(beta, covar) * self.baseline._ichf(params0, v, *args)

    def mean(self, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        if issubclass(self.baseline.__class__, ParametricLifetimeDistribution):
            return self._g(self.beta, covar) * self.baseline.mean(*args)
        else:
            super().mean(covar, *args)

    def mrl(self, t: np.ndarray, covar: np.ndarray, *args: np.ndarray) -> np.ndarray:
        if issubclass(self.baseline.__class__, ParametricLifetimeDistribution):
            t0 = t / self._g(self.beta, covar)
            return self._g(self.beta, covar) * self.baseline.mrl(t0, *args)
        else:
            super().mrl(t, covar, *args)


class ProportionalHazards(Regression):
    r"""Parametric Proportional Hazards Regression Model.

    The cumulative hazard function :math:`H` is linked to the multiplier
    function :math:`g` by the relation:

    .. math::

        H(t, x) = g(\beta, x) H_0(t) = e^{\beta \cdot x} H_0(t)

    where :math:`x` is a vector of covariates, :math:`\beta` is the coefficient
    vector of the effect of covariates, :math:`H_0` is the baseline cumulative
    hazard function.

    References
    ----------
    .. [1] Sun, J. (2006). The statistical analysis of interval-censored failure
        time data (Vol. 3, No. 1). New York: springer.
    """

    def _chf(
        self, params: np.ndarray, t: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        beta, params0 = self._dispatch(params)
        return self._g(beta, covar) * self.baseline._chf(params0, t, *args)

    def _hf(
        self, params: np.ndarray, t: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        beta, params0 = self._dispatch(params)
        return self._g(beta, covar) * self.baseline._hf(params0, t, *args)

    def _dhf(
        self, params: np.ndarray, t: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        beta, params0 = self._dispatch(params)
        return self._g(beta, covar) * self.baseline._dhf(params0, t, *args)

    def _jac_chf(
        self, params: np.ndarray, t: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        beta, params0 = self._dispatch(params)
        return np.column_stack(
            (
                self._jac_g(beta, covar) * self.baseline._chf(params0, t, *args),
                self._g(beta, covar) * self.baseline._jac_chf(params0, t, *args),
            )
        )

    def _jac_hf(
        self, params: np.ndarray, t: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        beta, params0 = self._dispatch(params)
        return np.column_stack(
            (
                self._jac_g(beta, covar) * self.baseline._hf(params0, t, *args),
                self._g(beta, covar) * self.baseline._jac_hf(params0, t, *args),
            )
        )

    def _ichf(
        self, params: np.ndarray, v: np.ndarray, covar: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        beta, params0 = self._dispatch(params)
        return self.baseline._ichf(params0, v / self._g(beta, covar), *args)
