from typing import Tuple, TYPE_CHECKING, Any

import numpy as np
from scipy import linalg
from scipy.stats import chi2

# TODO: needed but don't work
#if TYPE_CHECKING:
#from relife.lifetime_model._base import FittableParametricLifetimeModel

__all__ = ["wald_test", "scores_test"]

# TODO: add tests

def wald_test(model, c: np.ndarray = None, **kwargs) -> Tuple[float, float]:
    """Perform Wald's test (testing nullity of covariate effect)

    Args:
        model (FittableParametricLifetimeModel): model object representing a fitted model
        c (np.ndarray, optional): combination vector of 0 (beta is 0) and 1 (beta is not 0) indicating
            which covar coordinate is 0 in the null hypothesis
            Defaults to None, then the null hypothesis corresponds to null effect of all covariates
        **kwargs: to pass information or covariance matrix if fitting_results have not been yet computed
                    and stored into model (precisely in the case when wald_test is called during fitting_results
                    processing)

    Returns:
        Tuple[float, float]: test value and its corresponding pvalue

    """
    assert model.params is not None, "model has to be fitted before calling wald_test"
    if isinstance(c, list):
        assert model.nb_params == len(c)
        c = np.array(c)
    elif isinstance(c, np.ndarray):
        assert len(c.shape) == 1
        assert model.nb_params == c.shape[-1]

    if c is None and "information_matrix" in kwargs:
        information_matrix = kwargs["information_matrix"]
        covariance_matrix = None
    elif c is not None and "covariance_matrix" in kwargs:
        information_matrix = None
        covariance_matrix = kwargs["covariance_matrix"]
    else:
        information_matrix = model.fitting_results.information_matrix
        covariance_matrix = model.fitting_results.covariance_matrix

    if c is None:
        # null hypothesis is beta = 0
        ch2 = np.dot(model.params, np.dot(information_matrix, model.params))
        pval = chi2.sf(ch2, df=model.nb_params)
        return round(ch2, 6), round(pval, 6)
    else:
        # local test
        other_covar = np.where(c == 0)[0]

        ch2 = np.dot(
            model.params[other_covar],
            np.dot(
                linalg.inv(
                    covariance_matrix[np.ix_(other_covar, other_covar)]
                ),
                model.params[other_covar],
            ),
        )
        pval = chi2.sf(ch2, df=len(other_covar))
        return round(ch2, 6), round(pval, 6)


def scores_test(model, c: np.ndarray = None, *args, **kwargs) -> Tuple[float, float]:
    """Perform scores test (testing nullity of covariate effect)

    Args:
        model (FittableParametricLifetimeModel): model object representing a fitted model
        c (np.ndarray, optional): combination vector of 0 (beta is 0) and 1 (beta is not 0) indicating
            which covar coordinate is 0 in the null hypothesis
            Defaults to None, then the null hypothesis corresponds to null effect of all covariates

    Returns:
        Tuple[float, float]: test value and its corresponding pvalue

    """
    assert model.params is not None, "model has to be fitted before calling scores_test"
    if isinstance(c, list):
        assert model.nb_params == len(c)
        c = np.array(c)
    elif isinstance(c, np.ndarray):
        assert len(c.shape) == 1
        assert model.nb_params == c.shape[-1]
    #assert hasattr(model.fitting_results, "likelihood"), "you need likelihood object from fit to perform such a test"
    likelihood = model.fitting_results.likelihood
    # TODO: Réintroduire l'ancien approx_hessian scheme qui retournait une fonction de param ?
    #       Should it be introduced directly in Likelihood and usable during the fit as well ?
    assert hasattr(likelihood, "hess_negative_log"), "you need hess_negative_log to perform such a test"

    if c is None:
        # null hypothesis is beta = 0
        ch2 = np.dot(
            -likelihood.jac_negative_log(np.zeros(model.nb_params)),
            np.dot(
                linalg.inv(likelihood.hess_negative_log(np.zeros(model.nb_params))),
                -likelihood.jac_negative_log(np.zeros(model.nb_params)),
            ),
        )
        pval = chi2.sf(ch2, df=model.nb_params)
        return round(ch2, 3), round(pval, 3)
    else:
        # local test
        tested_covar = np.where(c != 0)[0]
        other_covar = np.where(c == 0)[0]

        # set seed for reproductibility
        model_under_h0 = model.__class__() # TODO: what about args ? Should we use something like sklearn clone ?


        model_under_h0.fit(  # TODO: fit_from_interval_censored_lifetimes and IntervalLifetimeLikelihood
            time=likelihood.time,
            covar=likelihood.covar[:, tested_covar],
            event=likelihood.event,
            entry=likelihood.entry,
            *args, **kwargs
        )
        beta_under_h0 = np.zeros(model.nb_params)
        beta_under_h0[tested_covar] = model_under_h0.params

        ch2 = np.dot(
            likelihood.jac_negative_log(beta_under_h0)[other_covar],
            np.dot(
                linalg.inv(likelihood.hess_negative_log(beta_under_h0))[np.ix_(other_covar, other_covar)],
                likelihood.jac_negative_log(beta_under_h0)[other_covar],
            ),
        )
        pval = chi2.sf(ch2, df=len(other_covar))
        return round(ch2, 6), round(pval, 6)
