import numpy as np
from scipy.optimize import Bounds

from relife.lifetime_model.regression import _CovarEffect
from relife.likelihood._lifetime_likelihood import PartialLifetimeLikelihood


class CoxBaseline:
    """
    Class for Cox non-parametric baseline
    """

    def hf(self, *args, kwargs):
        raise NotImplementedError

    def chf(self, *args, kwargs):
        raise NotImplementedError


class Cox:
    """
    Class for Cox, semi-parametric, Proportional Hazards, model
    """

    def __init__(self, coefficients=(None,)):
        self.covar_effect = _CovarEffect(coefficients)
        self.baseline = CoxBaseline()

    @property
    def params(self):
        return self.covar_effect.params

    @params.setter
    def params(self, value):
        self.covar_effect.params = value

    @property
    def nb_params(self):
        return self.covar_effect.nb_params

    def hf(self, covar, *args, kwargs):
        """
        The hazard function.

        Parameters
        ----------
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).

        TODO: update with CoxBaseline.hf args
        """
        return self.covar_effect.g(covar) * self.baseline.hf(*args, kwargs)

    def chf(self, covar, *args, kwargs):
        """
        The cumulative hazard function.

        Parameters
        ----------
        covar : float or np.ndarray
            Covariates values. float can only be valid if the regression has one coefficients.
            Otherwise it must be a ndarray of shape (nb_coef,) or (m, nb_coef)

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).

        TODO: update with CoxBaseline.chf args
        """
        return self.covar_effect.g(covar) * self.baseline.chf(*args, kwargs)

    def _get_initial_params(
            self, time, covar, event=None, entry=None
    ):
        self.covar_effect = _CovarEffect(
            (None,) * np.atleast_2d(np.asarray(covar)).shape[-1]
        )  # changes params structure depending on number of covar
        param0 = np.zeros_like(self.params, dtype=np.float64)
        return param0

    def _get_params_bounds(self):
        lb = np.full(self.nb_params, -np.inf),
        ub = np.full(self.nb_params, np.inf),
        return Bounds(lb, ub)

    def fit(
        self,
        time,
        covar,
        event=None,
        entry=None,
        optimizer_options=None,
    ):
        likelihood = PartialLifetimeLikelihood(
            self, time, covar, event=event, entry=entry
        )
        if optimizer_options is None:
            optimizer_options = {}
        if "bounds" not in optimizer_options:
            optimizer_options["bounds"] = self._get_params_bounds()
        fitting_results = likelihood.maximum_likelihood_estimation(**optimizer_options)
        self.params = fitting_results.optimal_params
        self.fitting_results = fitting_results
        return self