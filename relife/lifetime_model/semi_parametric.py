from regression import _CovarEffect # TODO: que fait-on du "_" ?


class CoxBaseline:
    """
    Class for Cox non-parametric baseline

    TODO: Be there or in non_parametric.py ?
    """

    def hf(self, *args, kwargs):
        raise NotImplementedError

    def chf(self, *args, kwargs):
        raise NotImplementedError


class Cox:
    """
    Class for Cox, semi-parametric, Proportional Hazards, model

    TODO: - Couldn't it be a LifetimeModel as non_parametric.py could be too,
            and ParametricLifetimeModel inherited from it and ParametricModel ?
          - It is not a ParametricModel, self._params does not exist,
            and is not fed with those of other ParametricModel attributes (covar_effect)
    """

    def __init__(self, coefficients=(None,)):
        self.covar_effect = _CovarEffect(coefficients)
        self.baseline = CoxBaseline()

    @property
    def params(self): # TODO: defined as LifetimeRegression.coefficients which is nowhere used
        return self.covar_effect.params

    @params.setter
    def params(self, value):
        self.covar_effect.params = value

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