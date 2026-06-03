# pyright: basic

import numpy as np
import pytest

from relife.datasets import load_insulator_string, load_power_transformer
from relife.lifetime_models import (
    Exponential,
    Gamma,
    Gompertz,
    LogLogistic,
    ParametricAcceleratedFailureTime,
    ParametricProportionalHazard,
    Weibull,
)

#######################################################################################
# DATA FIXTURES
#######################################################################################


@pytest.fixture
def power_transformer_data():
    return load_power_transformer()


@pytest.fixture
def insulator_string_data():
    return load_insulator_string()


#######################################################################################
# LIFETIME MODEL FIXTURES
#######################################################################################


_DISTRIBUTIONS = [
    Exponential(0.00795203),
    Weibull(3.46597395, 0.01227849),
    Gompertz(0.00865741, 0.06062632),
    Gamma(5.3571091, 0.06622822),
    LogLogistic(3.92614064, 0.0133325),
]


_COEFFICIENTS = (np.log(2), np.log(2))


_REGRESSIONS = [
    ParametricProportionalHazard(d, _COEFFICIENTS) for d in _DISTRIBUTIONS
] + [ParametricAcceleratedFailureTime(d, _COEFFICIENTS) for d in _DISTRIBUTIONS]


@pytest.fixture(params=_DISTRIBUTIONS, ids=[repr(d) for d in _DISTRIBUTIONS])
def distribution(request):
    yield request.param


@pytest.fixture(params=_REGRESSIONS, ids=[repr(r) for r in _REGRESSIONS])
def regression(request):
    yield request.param


#######################################################################################
# LIFETIME LIKELIHOOD FIXTURES
#######################################################################################


@pytest.fixture
def regression_likelihood(regression, insulator_string_data):
    covar = np.column_stack(
        (
            insulator_string_data["pHCl"],
            insulator_string_data["pH2SO4"],
        )
    )
    return regression.init_likelihood(
        insulator_string_data["time"],
        covar,
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
    )


#######################################################################################
# FROZEN LIFETIME FIXTURES
#######################################################################################

NB_ASSETS = 3


@pytest.fixture
def frozen_regression(regression):
    nb_coef = regression.covar_effect.get_params().size
    covar = np.linspace(0.0, 0.5, num=NB_ASSETS * nb_coef).reshape(NB_ASSETS, nb_coef)
    return regression.freeze(covar)


# @pytest.fixture
# def frozen_ar_distribution(distribution):
#     ar = distribution.isf(0.75)
#     return AgeReplacementModel(distribution).freeze(ar)
#
#
# @pytest.fixture
# def frozen_ar_regression(regression):
#     nb_coef = regression.covar_effect.get_params().size
#     covar = np.linspace(0.0, 0.5, num=NB_ASSETS * nb_coef).reshape(NB_ASSETS, nb_coef)
#     ar = regression.isf(0.75, covar)
#     return AgeReplacementModel(regression).freeze(ar, covar)


#######################################################################################
# ECONOMIC FIXTURES
#######################################################################################


# @pytest.fixture(
#     params=[
#         np.ones((), dtype=np.float64),
#         # np.ones((1,), dtype=np.float64),
#         np.ones((M,), dtype=np.float64),
#         # np.ones((M, 1), dtype=np.float64),
#     ],
#     ids=lambda cp: f"cp:{cp.shape}",
# )
# def cp(request):
#     return request.param
#
#
# # M = 3
# CF_RANGE = [5, 10, 20, 100, 1000]
#
#
# @pytest.fixture(
#     params=[
#         np.array(CF_RANGE[0], dtype=np.float64),  # ()
#         # np.array([CF_RANGE[0]], dtype=np.float64), # (1,)
#         np.array(CF_RANGE[:M], dtype=np.float64),  # (M,)
#         # np.array(CF_RANGE[:M], dtype=np.float64).reshape(-1, 1), # (M, 1)
#     ],
#     ids=lambda cf: f"cf:{cf.shape}",
# )
# def cf(request):
#     return request.param


@pytest.fixture(params=[0.0, 0.04], ids=lambda rate: f"discounting_rate:{rate}")
def discounting_rate(request):
    return request.param
