from typing import Tuple, TYPE_CHECKING

import numpy as np
from scipy import linalg
from scipy.stats import chi2

#if TYPE_CHECKING:
from relife.lifetime_model._base import FittableParametricLifetimeModel


def wald_test(model: FittableParametricLifetimeModel, c: np.ndarray = None) -> Tuple[float, float]:
    """Perform Wald's test (testing nullity of covariate effect)

    Args:
        model (FittableParametricLifetimeModel): model object representing a fitted model
        c (np.ndarray, optional): combination vector of 0 (beta is 0) and 1 (beta is not 0) indicating
            which covar coordinate is 0 in the null hypothesis
            Defaults to None, then the null hypothesis corresponds to null effect of all covariates

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

    if c is None:
        # null hypothesis is beta = 0
        ch2 = np.dot(model.params, np.dot(model.fitting_results.information_matrix, model.params))
        pval = chi2.sf(ch2, df=model.nb_params)
        return round(ch2, 6), round(pval, 6)
    else:
        # local test
        other_covar = np.where(c == 0)[0]

        ch2 = np.dot(
            model.params[other_covar],
            np.dot(
                linalg.inv(
                    model.fitting_results.covariance_matrix[np.ix_(other_covar, other_covar)] # TODO: why not use information_matrix ?
                ),
                model.params[other_covar],
            ),
        )
        pval = chi2.sf(ch2, df=len(other_covar))
        return round(ch2, 6), round(pval, 6)


def scores_test(model: FittableParametricLifetimeModel, c: np.ndarray = None, *args, **kwargs) -> Tuple[float, float]:
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
    assert hasattr(model.fitting_results, "likelihood"), "you need likelihood object from fit to perform such a test"
    likelihood = model.fitting_results.likelihood

    if c is None:
        # null hypothesis is beta = 0
        ch2 = np.dot(
            -likelihood.jac_negative_log(np.zeros(model.nb_params)),
            np.dot(
                model.fitting_results.covariance_matrix(np.zeros(model.nb_params)),
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
        model_under_h0 = model.__class__()
        model_under_h0.fit(
            time=likelihood.time,
            covar=likelihood.covar[:, tested_covar],
            event=likelihood.event,
            entry=likelihood.entry,
            *args, **kwargs
        )
        beta_under_h0 = np.zeros(model.nb_params)
        beta_under_h0[tested_covar] = model_under_h0.param

        ch2 = np.dot(
            likelihood.jac_negative_log(beta_under_h0)[other_covar],
            np.dot(
                model.fitting_results.covariance_matrix(beta_under_h0)[np.ix_(other_covar, other_covar)],
                likelihood.jac_negative_log(beta_under_h0)[other_covar],
            ),
        )
        pval = chi2.sf(ch2, df=len(other_covar))
        return round(ch2, 6), round(pval, 6)
