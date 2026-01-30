from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds
from scipy.stats import norm

from relife.lifetime_model._regression import CovarEffect
from relife.likelihood import PartialLifetimeLikelihood
from relife.likelihood._base import SCIPY_MINIMIZE_ORDER_2_ALGO, SCIPY_MINIMIZE_BOUND_ALGO


class BreslowBaseline:
    """
    Class for Cox non-parametric Breslow baseline
    """

    def __init__(
            self,
            covar_effect: CovarEffect,
            event_count: np.ndarray,
            ordered_event_covar: np.ndarray,
            psi: Callable,
    ):
        self.covar_effect = covar_effect
        self.event_count = event_count
        self.ordered_event_covar = ordered_event_covar
        self.psi = psi

    def chf(
        self, conf_int: bool = False, kp: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Knowing estimates of beta, computes the cumulative baseline hazard rate estimator and its confidence interval (optional)

        Args:
            conf_int (bool, optional): If true returns estimated confidence interval. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray] or np.ndarray: values of chf0 estimator and its confidence interval
            at 95% level. Arrays are of size :math:`m`
        """
        if kp:
            values = np.cumsum(
                1
                - (
                    1
                    - (
                        self.covar_effect.g(self.ordered_event_covar)
                        / self.psi()
                    )
                )
                ** (self.covar_effect.g(self.ordered_event_covar))
            )
        else:
            values = np.cumsum(self.event_count[:, None] / self.psi())
        if conf_int:
            var = np.cumsum(self.event_count[:, None] / self.psi() ** 2)
            conf_int = np.hstack(
                [
                    values[:, None]
                    + np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                    values[:, None]
                    - np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                ]
            )
            return values, conf_int
        else:
            return values

    def sf(self, conf_int: bool = False) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Knowing estimates of beta, computes the baseline survival function and its confidence interval (optional)

        Args:
            conf_int (bool, optional): If true returns estimated confidence interval. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray] or np.ndarray: values of chf0 estimator and its confidence interval
            at 95% level. Arrays are of size :math:`m`
        """
        if conf_int:
            chf, chf_conf_int = -self.chf(conf_int=True)
            return np.exp(-chf), np.exp(-chf_conf_int)
        else:
            return np.exp(-self.chf())


class Cox:
    """
    Class for Cox, semi-parametric, Proportional Hazards, model
    """

    def __init__(self, coefficients=(None,), baseline_estimator: str = "Breslow"):
        self.covar_effect = CovarEffect(coefficients)
        assert baseline_estimator == "Breslow", "The only Cox baseline estimator available is Breslow"
        self.baseline_estimator = baseline_estimator
        self._baseline = None

    @property
    def params(self):
        return self.covar_effect.params

    @params.setter
    def params(self, value):
        self.covar_effect.params = value

    @property
    def nb_params(self):
        return self.covar_effect.nb_params

    @property
    def baseline(self):
        if self._baseline is None:
            raise ValueError("Cox baseline is not available before model fitting")
        else:
            return self._baseline

    @baseline.setter
    def baseline(self, value):
        if not isinstance(value, BreslowBaseline):
            raise TypeError("Cox baseline must be a BreslowBaseline object")
        self._baseline = value

    def sf(
            self, covar: np.ndarray, conf_int: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Knowing estimates of beta, computes the sf estimator and confidence interval (optional)

        Args:
            covar (np.ndarray): one vector of covariate values, shape p
            conf_int (bool, optional): If true returns estimated confidence interval. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray] or np.ndarray:  values of sf estimator and its confidence interval
            at 95% level. Arrays are of size :math:`m`
        """
        values = self.baseline.sf() ** self.covar_effect.g(covar)
        if conf_int:
            psi = self.baseline.psi()
            psi_order_1 = self.baseline.psi(order=1)
            d_j_on_psi = self.baseline.event_count[:, None] / psi

            q3 = np.cumsum((psi_order_1 / psi - covar) * d_j_on_psi, axis=0)  # [m, p]
            q2 = np.squeeze(
                np.matmul(
                    q3[:, None, :],
                    np.matmul(self.fitting_results.covariance_matrix[None, :, :], q3[:, :, None]),
                )
            )  # m
            q1 = np.cumsum(d_j_on_psi * (1 / psi))

            var = (values ** 2) * (q1 + q2)

            conf_int = np.hstack(
                [
                    values[:, None]
                    + np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                    values[:, None]
                    - np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                ]
            )

            return values, conf_int
        else:
            return values

    def get_initial_params(
        self,
        time: NDArray[np.float64],
        covar: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        param0 = np.zeros_like(self.params, dtype=np.float64)
        return param0

    @property
    def params_bounds(self):
        lb = np.full(self.nb_params, -np.inf)
        ub = np.full(self.nb_params, np.inf)
        return Bounds(lb, ub)

    def fit(
        self,
        time,
        covar,
        event=None,
        entry=None,
        optimizer_options=None,
        seed: int = 1
    ):
        self.covar_effect = CovarEffect(
            (None,) * np.atleast_2d(np.asarray(covar, dtype=np.float64)).shape[-1]
        )  # changes params structure depending on number of covar

        likelihood = PartialLifetimeLikelihood(
            self, time, covar, event=event, entry=entry
        )

        if optimizer_options is None:
            optimizer_options = {}
        if "method" not in optimizer_options:
            optimizer_options["method"] = "trust-exact"
        if (optimizer_options["method"] in SCIPY_MINIMIZE_BOUND_ALGO) and ("bounds" not in optimizer_options):
            optimizer_options["bounds"] = self.params_bounds
        if (optimizer_options["method"] in SCIPY_MINIMIZE_ORDER_2_ALGO) and ("hess" not in optimizer_options):
            optimizer_options["hess"] = likelihood.hess_negative_log
        if "x0" not in optimizer_options:
            np.random.seed(seed)
            optimizer_options["x0"] = np.random.random(covar.shape[1])

        fitting_results = likelihood.maximum_likelihood_estimation(**optimizer_options)

        self.params = fitting_results.optimal_params
        self.fitting_results = fitting_results
        self.baseline = BreslowBaseline(
            covar_effect=self.covar_effect,
            event_count=likelihood._event_count,
            ordered_event_covar=likelihood._ordered_event_covar,
            psi=likelihood._psi,
        )

        return self