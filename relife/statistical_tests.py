#from typing import TYPE_CHECKING
from inspect import getfullargspec
import warnings

import numpy as np
from scipy import linalg
from scipy.stats import chi2
import matplotlib.pyplot as plt

# TODO: needed but don't work
#if TYPE_CHECKING:
#from relife.lifetime_model._base import FittableParametricLifetimeModel
from relife.likelihood._base import Likelihood
from relife.typing import ScipyMinimizeOptions

from relife.likelihood import IntervalLifetimeLikelihood
from relife.utils import is_2d_np_array, reshape_1d_arg, nearest_1dinterp, get_ordered_event_time

__all__ = ["wald_test", "scores_test"]

# TODO: add tests


def _try_fit_a_regression_model_from_likelihood_stored_data(
        model,
        likelihood: Likelihood,
        tested_covar: np.ndarray = None,
        is_tested_covar_index: bool = True,
        value_index: np.ndarray = None,
        **kwargs
):
    """
    'model' is assumed to be an unfitted (clone from another) model

    'likelihood' is assumed to come from (that) another model.fitting_results
    and to store the fitting data we are going to extract to fit model

    'tested_covar' is a subset of the original covar

    'is_tested_covar_index' is tested_covar np.array of int indicating index of
    covar of interest or np.array of float giving direct values

    'kwargs' args not stored into likelihood (e.g. optimizer_options)
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
            # TODO: The following may assume those expected fit args found in likelihood restrict to time, covar/model_args, event, entry
            #       There also may be unreliable assumptions on model_args.
            #       I don't like working with so much indeterminacy, we should make things perfectly clear.
            #       model_args is very difficult to work with...
            if tested_covar is not None:
                if args in ["model_args", "covar"]: # "covar" for Cox that does not have model_args (including covar) yet
                    assert fit_args[args] is not None, "You need covar data to fit your model"
                    # The following assumes that all 2d np.array in model_args have covar
                    # representing columns and must be sliced the same way...
                    if is_2d_np_array(fit_args[args]):
                        # if model_args is 2d np.array, one assumes it is covar
                        fit_args[args] = fit_args[args][:, tested_covar] if is_tested_covar_index else tested_covar
                    elif isinstance(fit_args[args], tuple):
                        assert is_tested_covar_index, "cannot replace covar by tested_covar blindly"
                        fit_args[args] = tuple(
                            ma[:, tested_covar] if is_2d_np_array(ma) else ma for ma in fit_args[args]
                    )
            if value_index is not None:
                # The following assumes that all np.ndarray found in fit args and likelihood (possibly in model_args)
                # have a sample dimension to be filtered...
                if isinstance(fit_args[args], np.ndarray):
                    fit_args[args] = reshape_1d_arg(fit_args[args])[value_index, :]
                elif isinstance(fit_args[args], tuple):
                        fit_args[args] = tuple(
                            reshape_1d_arg(ma)[value_index, :] if isinstance(ma, np.ndarray) else ma for ma in fit_args[args]
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


def wald_test(model, c: np.ndarray = None, **kwargs) -> tuple[float, float]:
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
        tuple[float, float]: test value and its corresponding pvalue

    """
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
        assert hasattr(model, "fitting_results"), "model has to be fitted before calling wald_test"
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


def scores_test(model, model_init_kwargs: dict, c: np.ndarray = None, **kwargs) -> tuple[float, float]:
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
        **kwargs: args not stored into likelihood (e.g. optimizer_options) to be passed to
            _try_fit_a_regression_model_from_likelihood_stored_data

    Returns:
        tuple[float, float]: test value and its corresponding pvalue

    """
    assert hasattr(model, "fitting_results"), "model has to be fitted before calling scores_test"
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


def likelihood_ratio_test(model, model_init_kwargs: dict, c: np.ndarray = None, **kwargs) -> tuple[float, float]:
    """Perform likelihood ratio test (testing nullity of covariate effect)

    Args:
        model (FittableParametricLifetimeModel): model object representing a fitted model
        model_init_kwargs (dict): model.__init__ arguments to clone the model specification
        c (np.ndarray, optional): combination vector of 0 (beta is 0) and 1 (beta is not 0) indicating
            which covar coordinate is 0 in the null hypothesis. Defaults to None, then the null hypothesis
            corresponds to null effect of all covariates
        **kwargs: args not stored into likelihood (e.g. optimizer_options) to be passed to
            _try_fit_a_regression_model_from_likelihood_stored_data

    Returns:
        tuple[float, float]: test value and its corresponding pvalue

    """
    assert hasattr(model, "fitting_results"), "model has to be fitted before calling likelihood_ratio_test"
    if isinstance(c, list):
        assert model.nb_params == len(c)
        c = np.array(c)
    elif isinstance(c, np.ndarray):
        assert len(c.shape) == 1
        assert model.nb_params == c.shape[-1]

    likelihood = model.fitting_results.likelihood

    if c is None:
        # null hypothesis is beta = 0
        neg_pl_beta = likelihood.negative_log(model.params)
        neg_pl_beta_0 = likelihood.negative_log(np.zeros_like(model.params))
        ch2 = 2 * (neg_pl_beta_0 - neg_pl_beta)
        pval = chi2.sf(ch2, df=model.nb_params)
        return round(ch2, 6), round(pval, 6)
    else:
        # local test
        tested_covar = np.where(c != 0)[0]
        other_covar = np.where(c == 0)[0]

        neg_pl_beta = likelihood.negative_log(model.params)
        model_under_h0 = model.__class__(**model_init_kwargs)
        model_under_h0 = _try_fit_a_regression_model_from_likelihood_stored_data(
            model=model_under_h0, likelihood=likelihood, tested_covar=tested_covar, **kwargs
        )
        likelihood_under_h0 = model_under_h0.fitting_results.likelihood
        neg_pl_beta_under_h0 = likelihood_under_h0.negative_log(
            model_under_h0.params
        )
        ch2 = 2 * (neg_pl_beta_under_h0 - neg_pl_beta)
        pval = chi2.sf(ch2, df=len(other_covar))
        return round(ch2, 6), round(pval, 6)


def proportionality_effect(
    model,
    model_init_kwargs: dict,
    tested_covar: np.ndarray,
    nb_strata: int = 4,
    andersen: bool = False,
    is_categorical: bool = False,
    plot: bool = False,
    **kwargs
) -> None | np.ndarray:
    """Graphical checks of the proportional effects of covariates assumption
    # TODO: ne marche pas car on essaie de fitter sur une covar constante (après discrétisation et stratification)
           reboucler avec William pour faire le point

    Args:
        model (ProportionalHazard or Cox): instance model
        model_init_kwargs (dict): model.__init__ arguments to clone the model specification
        tested_covar (np.ndarray) : covar values on which the proportionality effect is tested
        nb_strata (int, optional): number of strata used for covariate values. Defaults to 4.
        andersen (bool, optional): If True, Andersen plots are used. Defaults to False, then difference of log cumulative hazard rates is used
        is_categorical (bool, optional): If True, covar values are considered like categories and are not transformed
        plot (bool, optional): If True, plot the graphical check of the proportionality effect
        **kwargs: args not stored into likelihood (e.g. optimizer_options) to be passed to
            _try_fit_a_regression_model_from_likelihood_stored_data
    Raises:
        ValueError: the number of strata must not be too high to keep enough data per stratum
    """
    assert len(tested_covar.shape) == 1, f"covar has shape {tested_covar.shape} but must be 1d"
    assert hasattr(model, "fitting_results"), "model has to be fitted before calling proportionality_effect"

    likelihood = model.fitting_results.likelihood
    timeline = np.sort(likelihood.time)

    # if not categorical covar, encode it
    if not is_categorical:
        bins = np.quantile(tested_covar, q=np.cumsum(np.ones(nb_strata) / nb_strata))
        tested_covar = np.digitize(tested_covar, bins, right=True) + 1
        covar_strata_values = np.unique(tested_covar)
        nb_cat_values = len(covar_strata_values)
    else:
        covar_strata_values = np.unique(tested_covar)
        nb_cat_values = len(covar_strata_values)
        if nb_strata != nb_cat_values:
            raise ValueError(
                f"If covar is categorical, nb strata must equal nb of categorical values ({nb_strata} nb strata != {nb_cat_values})"
            )

    chf0_strata = np.empty((nb_cat_values, len(timeline)))

    # compute chf0 for each stratum of covar value
    for i, value in enumerate(covar_strata_values):
        value_index = np.where(tested_covar == value)[0]
        if len(value_index) == 1:
            raise ValueError(
                f"Nb of strata is too high and {i}-th stratum only corresponds to one value. Decrease nb_strata value"
            )
        model_at_value = model.__class__(**model_init_kwargs)
        model_at_value = _try_fit_a_regression_model_from_likelihood_stored_data(
            model=model_at_value, likelihood=likelihood, tested_covar=tested_covar,
            is_tested_covar_index=False, value_index=value_index, **kwargs
        )
        likelihood_at_value = model_at_value.fitting_results.likelihood
        ordered_event_time_at_value, _, _ = get_ordered_event_time(
            time=likelihood_at_value.time, event=likelihood_at_value.event
        )
        chf0_strata[i] = nearest_1dinterp(
            timeline, ordered_event_time_at_value, model_at_value.baseline.chf()
        )

    if andersen:
        if plot:
            # set figure grid
            fig, ax = plt.subplots()
            for i in range(1, chf0_strata.shape[0]):
                ax.step(
                    chf0_strata[0],
                    chf0_strata[i],
                    where="post",
                    label=f"strata {i + 1} vs. strata 1",
                )
                ax.set_xlabel("Cumulated baseline hazard rate on stratum 1")
                ax.set_ylabel("Cumulated baseline hazard rate on stratum K")
                ax.legend()
            plt.show()
        else:
            return chf0_strata

    else:
        log_chf0_diff = np.log(
            chf0_strata[1:] / np.full_like(chf0_strata[1:], chf0_strata[0])
        )
        if plot:
            # set figure grid
            fig, ax = plt.subplots()
            for i in range(log_chf0_diff.shape[0]):
                ax.step(
                    timeline,
                    log_chf0_diff[i],
                    where="post",
                    label=f"log (strata {i + 2} / strata 1)",
                )
                ax.set_xlabel("Time on study")
                ax.set_ylabel(
                    "Difference in log cumulative hazard rates strata"
                )
                ax.legend()
            plt.show()
        else:
            return log_chf0_diff