import numpy as np
import pytest

from relife.data import load_power_transformer, load_insulator_string
from relife.lifetime_model import (
    Exponential,
    Weibull,
    LogLogistic,
    Gamma,
    Gompertz,
    ProportionalHazard,
    AcceleratedFailureTime, EquilibriumDistribution,
)


@pytest.fixture(
    params=[
        Exponential(0.00795203),
        Weibull(3.46597395, 0.01227849),
        Gompertz(0.00865741, 0.06062632),
        Gamma(5.3571091, 0.06622822),
        LogLogistic(3.92614064, 0.0133325),
    ],
    ids=["exponential", "weibull", "gompertz", "gamma", "loglogistic"]
)
def distribution(request):
    return request.param


@pytest.fixture
def equilibrium_distribution(distribution):
    return EquilibriumDistribution(distribution)


@pytest.fixture(
    params=[
        ProportionalHazard,
        AcceleratedFailureTime,
    ],
)
def regression(request, distribution):
    return request.param(distribution, (0.1,) * 3)



@pytest.fixture(
    params=[
        # np.float64(1),
        # np.ones((1,), dtype=np.float64),
        # np.ones((3,), dtype=np.float64),
        # np.ones((1, 1), dtype=np.float64),
        # np.ones((3, 1), dtype=np.float64),
        # np.ones((1, 3), dtype=np.float64),
        np.ones((10, 3), dtype=np.float64),
    ],
    ids=[
        # "time()",
        # "time(1,)",
        # "time(3,)",
        # "time(1,1)",
        # "time(3,1)",
        # "time(1,3)",
        "time(10, 3)"
]
)
def time(request):
    return request.param


@pytest.fixture(
    params=[
        np.float64(0.5),
        np.ones((1,), dtype=np.float64)*0.5,
        np.ones((3,), dtype=np.float64)*0.5,
        np.ones((1, 1), dtype=np.float64)*0.5,
        np.ones((3, 1), dtype=np.float64)*0.5,
        np.ones((1, 3), dtype=np.float64)*0.5,
        np.ones((10, 3), dtype=np.float64)*0.5,
    ],
    ids=["probability()", "probability(1,)", "probability(3,)", "probability(1,1)", "probability(3,1)", "probability(1,3)", "probability(10, 3)"]
)
def probability(request):
    return request.param

@pytest.fixture
def covar():
    def _covar(*d):
        return np.ones(d, dtype=np.float64)

    return _covar


@pytest.fixture
def ar():
    def _ar(n):
        return np.ones(n, dtype=np.float64)

    return _ar



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

@pytest.fixture
def insulator_string_data():
    return load_insulator_string()