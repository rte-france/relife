import pytest
import numpy as np
from relife.lifetime_model import Weibull, Gompertz, Gamma, LogLogistic
from relife.policy import OneCycleRunToFailurePolicy, RunToFailurePolicy


@pytest.fixture(
    params=[
        Weibull(2, 0.05),
        Gompertz(0.01, 0.1),
        Gamma(2, 0.05),
        LogLogistic(3, 0.05),
    ],
    ids=lambda distri: f"{distri.__class__.__name__}",
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
        # OneCycleRunToFailurePolicy,
        RunToFailurePolicy,
    ],
)
def run_to_failure_policy(request, distribution, cf, discounting_rate):
    return request.param(distribution, cf, discounting_rate=discounting_rate)