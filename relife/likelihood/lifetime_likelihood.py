from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize

from relife.data import LifetimeData

from . import FittingResults, approx_hessian
from ._base import Likelihood

if TYPE_CHECKING:
    from relife.lifetime_model import (
        LifetimeDistribution,
        LifetimeRegression,
        MinimumDistribution,
    )


class LikelihoodFromLifetimes(Likelihood):
    """
    Generic likelihood object for parametric_model lifetime model

    Parameters
    ----------
    model : ParametricLifetimeDistribution
        Underlying core used to compute probability functions
    lifetime_data : LifetimeData
        Observed lifetime data used one which the likelihood is evaluated
    """

    def __init__(
        self,
        model: Union[LifetimeDistribution, LifetimeRegression, MinimumDistribution],
        lifetime_data: LifetimeData,
    ):
        self.model = copy.deepcopy(model)
        self.data = lifetime_data

    def _complete_contribs(self, lifetime_data: LifetimeData) -> Optional[np.float64]:
        if lifetime_data.complete is None:
            return None
        return -np.sum(
            np.log(
                self.model.hf(
                    lifetime_data.complete.lifetime_values,
                    *lifetime_data.complete.args,
                )
            )  # (m, 1)
        )  # ()

    def _right_censored_contribs(self, lifetime_data: LifetimeData) -> Optional[np.float64]:
        if lifetime_data.complete_or_right_censored is None:
            return None
        return np.sum(
            self.model.chf(
                lifetime_data.complete_or_right_censored.lifetime_values,
                *lifetime_data.complete_or_right_censored.args,
            ),
            dtype=np.float64,  # (m, 1)
        )  # ()

    def _left_censored_contribs(self, lifetime_data: LifetimeData) -> Optional[np.float64]:
        if lifetime_data.left_censoring is None:
            return None
        return -np.sum(
            np.log(
                -np.expm1(
                    -self.model.chf(
                        lifetime_data.left_censoring.lifetime_values,
                        *lifetime_data.left_censoring.args,
                    )
                )
            )  # (m, 1)
        )  # ()

    def _left_truncations_contribs(self, lifetime_data: LifetimeData) -> Optional[np.float64]:
        if lifetime_data.left_truncation is None:
            return None
        return -np.sum(
            self.model.chf(
                lifetime_data.left_truncation.lifetime_values,
                *lifetime_data.left_truncation.args,
            ),  # (m, 1)
            dtype=np.float64,
        )  # ()

    def _jac_complete_contribs(self, lifetime_data: LifetimeData) -> Optional[NDArray[np.float64]]:
        if lifetime_data.complete is None:
            return None
        return -np.sum(
            self.model.jac_hf(
                lifetime_data.complete.lifetime_values,
                *lifetime_data.complete.args,
                asarray=True,
            )  # (p, m, 1)
            / self.model.hf(
                lifetime_data.complete.lifetime_values,
                *lifetime_data.complete.args,
            ),  # (m, 1)
            axis=(1, 2),
        )  # (p,)

    def _jac_right_censored_contribs(self, lifetime_data: LifetimeData) -> Optional[NDArray[np.float64]]:
        if lifetime_data.complete_or_right_censored is None:
            return None
        return np.sum(
            self.model.jac_chf(
                lifetime_data.complete_or_right_censored.lifetime_values,
                *lifetime_data.complete_or_right_censored.args,
                asarray=True,
            ),  # (p, m, 1)
            axis=(1, 2),
        )  # (p,)

    def _jac_left_censored_contribs(self, lifetime_data: LifetimeData) -> Optional[NDArray[np.float64]]:
        if lifetime_data.left_censoring is None:
            return None
        return -np.sum(
            self.model.jac_chf(
                lifetime_data.left_censoring.lifetime_values,
                *lifetime_data.left_censoring.args,
                asarray=True,
            )  # (p, m, 1)
            / np.expm1(
                self.model.chf(
                    lifetime_data.left_censoring.lifetime_values,
                    *lifetime_data.left_censoring.args,
                )
            ),  # (m, 1)
            axis=(1, 2),
        )  # (p,)

    def _jac_left_truncations_contribs(self, lifetime_data: LifetimeData) -> Optional[NDArray[np.float64]]:
        if lifetime_data.left_truncation is None:
            return None
        return -np.sum(
            self.model.jac_chf(
                lifetime_data.left_truncation.lifetime_values,
                *lifetime_data.left_truncation.args,
                asarray=True,
            ),  # (p, m, 1)
            axis=(1, 2),
        )  # (p,)

    def negative_log(
        self,
        params: NDArray[np.float64],  # (p,)
    ) -> np.float64:
        model_params = np.copy(self.model.params)
        self.model.params = params  # changes model params
        contributions = (
            self._complete_contribs(self.data),
            self._right_censored_contribs(self.data),
            self._left_censored_contribs(self.data),
            self._left_truncations_contribs(self.data),
        )
        self.model.params = model_params  # reset model params (negative_log must not change model params)
        return sum(x for x in contributions if x is not None)  # ()

    def jac_negative_log(
        self,
        params: NDArray[np.float64],  # (p,)
    ) -> NDArray[np.float64]:
        """
        Jacobian of the negative log likelihood.

        The jacobian (here gradient) is computed with respect to parameters

        Parameters
        ----------
        params : ndarray
            Parameters values on which the jacobian is evaluated

        Returns
        -------
        ndarray
            Jacobian of the negative log likelihood value
        """
        model_params = np.copy(self.model.params)
        self.model.params = params  # changes model params
        jac_contributions = (
            self._jac_complete_contribs(self.data),
            self._jac_right_censored_contribs(self.data),
            self._jac_left_censored_contribs(self.data),
            self._jac_left_truncations_contribs(self.data),
        )
        self.model.params = model_params  # reset model params (jac_negative_log must not change model params)
        return sum(x for x in jac_contributions if x is not None)  # (p,)

    def maximum_likelihood_estimation(self, **kwargs: Any) -> FittingResults:
        param0 = _init_params_values(self.model, self.data)

        # configure and run the optimizer
        minimize_kwargs = {
            "method": kwargs.get("method", "L-BFGS-B"),
            "constraints": kwargs.get("constraints", ()),
            "bounds": kwargs.get("bounds", _get_params_bounds(self.model)),
            "x0": kwargs.get("x0", param0),
        }
        optimizer = minimize(
            self.negative_log,
            minimize_kwargs.pop("x0"),
            jac=self.jac_negative_log,
            **minimize_kwargs,
        )
        optimal_params = np.copy(optimizer.x)
        neg_log_likelihood = np.copy(optimizer.fun)  # neg_log_likelihood value at optimal
        hessian = approx_hessian(self, optimal_params)
        covariance_matrix = np.linalg.inv(hessian)
        return FittingResults(
            self.data.nb_samples, optimal_params, neg_log_likelihood, covariance_matrix=covariance_matrix
        )


M = TypeVar("M", bound=Union["LifetimeDistribution", "LifetimeRegression", "MinimumDistribution"])


def _init_params_values(model: M, lifetime_data: LifetimeData) -> NDArray[np.float64]:
    from relife.lifetime_model import Gompertz, LifetimeRegression

    if isinstance(model, LifetimeRegression):
        model.baseline.params = _init_params_values(model.baseline, lifetime_data)
        param0 = np.zeros_like(model.params, dtype=np.float64)
        param0[-model.baseline.params.size :] = model.baseline.params
        return param0
    elif isinstance(model, Gompertz):
        param0 = np.empty(model.nb_params, dtype=np.float64)
        rate = np.pi / (np.sqrt(6) * np.std(lifetime_data.complete_or_right_censored.lifetime_values))
        shape = np.exp(-rate * np.mean(lifetime_data.complete_or_right_censored.lifetime_values))
        param0[0] = shape
        param0[1] = rate
        return param0
    else:  # other cases : Weibull, Gamma, Exponential, ...
        if lifetime_data.complete_or_right_censored is not None:
            param0 = np.ones(model.nb_params, dtype=np.float64)
            param0[-1] = 1 / np.median(lifetime_data.complete_or_right_censored.lifetime_values)
            return param0
        return np.zeros(model.nb_params, dtype=np.float64)


def _get_params_bounds(model: M) -> Bounds:
    from relife.lifetime_model import LifetimeRegression

    if isinstance(model, LifetimeRegression):
        lb = np.concatenate(
            (
                np.full(model.covar_effect.nb_params, -np.inf),
                _get_params_bounds(model.baseline).lb,
            )
        )
        ub = np.concatenate(
            (
                np.full(model.covar_effect.nb_params, np.inf),
                _get_params_bounds(model.baseline).ub,
            )
        )
        return Bounds(lb, ub)
    else:
        return Bounds(
            np.full(model.nb_params, np.finfo(float).resolution),
            np.full(model.nb_params, np.inf),
        )
