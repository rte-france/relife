from typing import Tuple, TYPE_CHECKING, Any
from inspect import getfullargspec
import warnings

import numpy as np
from scipy import linalg
from scipy.stats import chi2

# TODO: needed but don't work
#if TYPE_CHECKING:
#from relife.lifetime_model._base import FittableParametricLifetimeModel
from relife.likelihood._lifetime_likelihood import IntervalLifetimeLikelihood
from relife.typing import ScipyMinimizeOptions
from relife.utils._array_api import is_2d_np_array

__all__ = ["wald_test", "scores_test"]

# TODO: add tests


def _try_fit_a_regression_model_from_likelihood_stored_data(
        model,
        likelihood,
        tested_covar,
        **kwargs
):
    """
    'model' is assumed to be an unfitted (clone from another) model

    'likelihood' is assumed to come from (that) another model.fitting_results
    and to store the fitting data we are going to extract to fit model

    'tested_covar' is a subset of the original covar
    """
    if isinstance(likelihood, IntervalLifetimeLikelihood):
        fit_method = getattr(model, "fit_from_interval_censored_lifetimes")
    else:
        fit_method = getattr(model, "fit")

    fit_args_names = getfullargspec(fit_method)[0]

    fit_args = {}
    fit_args.update(kwargs)  # args not stored into likelihood (e.g. optimizer_options), will be override otherwise

    for args in fit_args_names:
        if args == "self":
            continue
        if hasattr(likelihood, args):
            fit_args[args] = getattr(likelihood, args)
            if args in ["model_args", "covar"]:
                assert fit_args[args] is not None, "You need covar data to fit your model"
                # The following assumes that all 2d_np array in model_args have covar
                # representing columns and must be sliced the same way...
                if is_2d_np_array(fit_args[args]):
                    fit_args[args] = fit_args[args][:, tested_covar]
                elif isinstance(fit_args[args], tuple):
                    fit_args[args] = tuple(
                        ma[:, tested_covar] if is_2d_np_array(ma) else ma for ma in fit_args[args]
                    )
        elif args in kwargs:
            pass
        else:
            warnings.warn(f"fit_method requires arg {args}, but is not available neither from likelihood object nor from kwargs")

    try:
        model = fit_method(**fit_args)
    except Exception as e:
        print(e)

    return model


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


def scores_test(model, model_init_kwargs: dict, c: np.ndarray = None, **kwargs) -> Tuple[float, float]:
    """Perform scores test (testing nullity of covariate effect)

    Args:
        model (FittableParametricLifetimeModel): model object representing a fitted model
        model_init_kwargs (dict): model.__init__ arguments to clone the model specification
        TODO: Should we use something like sklearn clone ? Make a issue for it ?
                This way, we could remove the dependence on model_init_kwargs, but it probably requires
                introducing some conventions (such as storing all Model.__init__ args as attributes).
                We did kinda the same thing when storing fitting data into Likelihood ...
        c (np.ndarray, optional): combination vector of 0 (beta is 0) and 1 (beta is not 0) indicating
            which covar coordinate is 0 in the null hypothesis
            Defaults to None, then the null hypothesis corresponds to null effect of all covariates
        optimizer_options (ScipyMinimizeOptions): fit optimizer options

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

    likelihood = model.fitting_results.likelihood

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
        model_under_h0 = model.__class__(**model_init_kwargs)
        model_under_h0 = _try_fit_a_regression_model_from_likelihood_stored_data(
            model=model_under_h0, likelihood=likelihood, tested_covar=tested_covar, **kwargs
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
