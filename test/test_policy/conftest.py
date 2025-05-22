import pytest
import numpy as np
from relife.lifetime_model import Weibull, Gompertz, Gamma, LogLogistic, Exponential
from relife.policy import OneCycleRunToFailurePolicy, RunToFailurePolicy, OneCycleAgeReplacementPolicy, \
    AgeReplacementPolicy


@pytest.fixture(
    params=[
        Weibull(3.46597395, 0.01227849),
        Gompertz(0.00865741, 0.06062632),
        Gamma(5.3571091, 0.06622822),
        LogLogistic(3.92614064, 0.0133325),
    ],
    ids=["Weibull", "Gompertz", "Gamma", "LogLogistic"],
)
def distribution(request):
    return request.param


@pytest.fixture(
    params=[0., 0.04],
    ids=lambda rate: f"discounting_rate:{rate}"
)
def discounting_rate(request):
    return request.param


@pytest.fixture
def cp():
    return 1.

@pytest.fixture
def cf(cp):
    return cp + np.array([5, 10, 20, 100, 1000]).reshape(-1, 1)

@pytest.fixture(
    params=[
        OneCycleRunToFailurePolicy,
        RunToFailurePolicy,
    ],
)
def run_to_failure_policy(request, distribution, cf, discounting_rate):
    return request.param(distribution, cf, discounting_rate=discounting_rate)

@pytest.fixture(
    params=[
        OneCycleAgeReplacementPolicy,
        AgeReplacementPolicy,
    ],
)
def age_replacement_policy(request, distribution, cf, cp, discounting_rate):
    return request.param(distribution, cf, cp, discounting_rate=discounting_rate)