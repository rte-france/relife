import numpy as np
import pytest

from relife.data import load_power_transformer
from relife.lifetime_model import (
    Exponential,
    Weibull,
    LogLogistic,
    Gamma,
    Gompertz,
    ProportionalHazard,
    AFT,
)

@pytest.fixture
def exponential():
    return Exponential(0.00795203)


@pytest.fixture
def weibull():
    return Weibull(3.46597395, 0.01227849)


@pytest.fixture
def gompertz():
    return Gompertz(0.00865741, 0.06062632)


@pytest.fixture
def gamma():
    return Gamma(5.3571091, 0.06622822)


@pytest.fixture
def loglogistic():
    return LogLogistic(3.92614064, 0.0133325)


@pytest.fixture
def distribution_map(exponential, weibull, gompertz, gamma, loglogistic):
    return {
        "exponential" : exponential,
        "weibull": weibull,
        "gompertz": gompertz,
        "gamma": gamma,
        "loglogistic": loglogistic,
    }


@pytest.fixture
def nb_coef():
    return 3




@pytest.fixture(
    params=[
        Exponential(0.00795203),
        Weibull(3.46597395, 0.01227849),
        Gompertz(0.00865741, 0.06062632),
        Gamma(5.3571091, 0.06622822),
        LogLogistic(3.92614064, 0.0133325),
    ],
    ids=["pph(exponential)", "pph(weibull)", "pph(gompertz)", "pph(gamma)", "pph(loglogistic)"],
)
def proportional_hazard(request, nb_coef):
    yield ProportionalHazard(request.param, coef=(0.1,) * nb_coef)


@pytest.fixture(
    params=[
        Exponential(0.00795203),
        Weibull(3.46597395, 0.01227849),
        Gompertz(0.00865741, 0.06062632),
        Gamma(5.3571091, 0.06622822),
        LogLogistic(3.92614064, 0.0133325),
    ],
    ids=["aft(exponential)", "aft(weibull)", "aft(gompertz)", "aft(gamma)", "aft(loglogistic)"],
)
def aft(request):
    yield AFT(request.param, coef=(0.1, 0.2, 0.3))


@pytest.fixture
def time():
    def _time(*d: int):
        if not bool(d):
            return 1.0
        return np.ones(d, dtype=np.float64)

    return _time


@pytest.fixture
def probability():
    def _probability(*d: int):
        if not bool(d):
            return 0.5
        return np.ones(d, dtype=np.float64) * 0.5

    return _probability


@pytest.fixture
def covar(nb_coef):
    def _covar(m):
        return np.ones((m, nb_coef), dtype=np.float64)

    return _covar


@pytest.fixture
def a():
    def _a(*d: int):
        if not bool(d):
            return 2.0
        return 2.0 * np.ones(d, dtype=np.float64)

    return _a


@pytest.fixture
def b():
    def _b(*d: int):
        if not bool(d):
            return 8.0
        return 8.0 * np.ones(d, dtype=np.float64)

    return _b



@pytest.fixture
def power_transformer_data():
    return load_power_transformer()
