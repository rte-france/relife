import numpy as np
import pytest

from relife.lifetime_model import Weibull, LogLogistic, Gamma, Gompertz, ProportionalHazard, AFT


@pytest.fixture(scope="package")
def weibull():
    return Weibull(2, 0.05)

@pytest.fixture(scope="package")
def gompertz():
    return Gompertz(0.01, 0.1)

@pytest.fixture(scope="package")
def gamma():
    return Gamma(2, 0.05)

@pytest.fixture(scope="package")
def loglogistic():
    return LogLogistic(3, 0.05)


@pytest.fixture(scope="package")
def nb_coef():
    return 3


@pytest.fixture(
    scope="package",
    params=[
        Weibull(2, 0.05),
        Gompertz(0.01, 0.1),
        Gamma(2, 0.05),
        LogLogistic(3, 0.05)
    ],
    ids=[
        "pph(weibull)",
        "pph(gompertz)",
        "pph(gamma)",
        "pph(loglogistic)"
    ]
)
def proportional_hazard(request, nb_coef):
    yield ProportionalHazard(request.param, coef=(0.1,)*nb_coef)


@pytest.fixture(
    scope="package",
    params=[
        Weibull(2, 0.05),
        Gompertz(0.01, 0.1),
        Gamma(2, 0.05),
        LogLogistic(3, 0.05)
    ],
    ids=[
        "aft(weibull)",
        "aft(gompertz)",
        "aft(gamma)",
        "aft(loglogistic)"
    ]
)
def aft(request):
    yield AFT(request.param, coef=(0.1, 0.2, 0.3))



@pytest.fixture(scope="module")
def time():
    def _time(shape = None):
        if shape is None:
            return 1.
        return np.ones(shape, dtype=np.float64)
    return _time


@pytest.fixture(scope="module")
def probability():
    def _probability(shape = None):
        if shape is None:
            return 0.5
        return np.ones(shape, dtype=np.float64)*0.5
    return _probability


@pytest.fixture(scope="module")
def covar(nb_coef):
    def _covar(m):
        return np.ones((m, nb_coef), dtype=np.float64)
    return _covar


