import numpy as np
import pytest

from relife.lifetime_model import Weibull, Exponential, Gompertz, Gamma, LogLogistic
from relife.stochastic_process import RenewalProcess

DISTRIBUTION_INSTANCES = [
    Exponential(0.00795203),
    Weibull(3.46597395, 0.01227849),
    Gompertz(0.00865741, 0.06062632),
    Gamma(5.3571091, 0.06622822),
    LogLogistic(3.92614064, 0.0133325),
]

@pytest.fixture(
    params=DISTRIBUTION_INSTANCES,
    ids=["Exponential", "Weibull", "Gompertz", "Gamma", "LogLogistic"],
)
def distribution(request):
    return request.param

# for distri in DISTRIBUTION_INSTANCES:
#     print(type(distri).__name__, distri.params)
#     renewal_process = RenewalProcess(distri)
#     expected_params = distri.params.copy()
#     q1 = distri.ppf(0.25)
#     q3 = distri.ppf(0.75)
#     lifetime_data = renewal_process.sample_lifetime_data(10 * q3, t0=q1, size=1000, seed=10)
#     print("nb lifetime data :", len(lifetime_data))
#     distri.fit_from_lifetime_data(lifetime_data)
#     print(distri.params)

def test_sample_lifetime_data(distribution):
    renewal_process = RenewalProcess(distribution)
    expected_params = distribution.params.copy()
    q1 = distribution.ppf(0.25)
    q3 = distribution.ppf(0.75)
    lifetime_data = renewal_process.sample_lifetime_data(10*q3, t0=q1, size=1000, seed=10)
    print("nb lifetime data :", len(lifetime_data))
    distribution.fit_from_lifetime_data(lifetime_data)
    print(distribution.params)
    assert np.allclose(distribution.params, expected_params, rtol=1e-3)

# def test():
#     for distri in DISTRIBUTION_INSTANCES:
#         print(type(distri).__name__, distri.params)
#         renewal_process = RenewalProcess(distri)
#         expected_params = distri.params.copy()
#         q1 = distri.ppf(0.25)
#         q3 = distri.ppf(0.75)
#         lifetime_data = renewal_process.sample_lifetime_data(10*q3, t0=q1, size=1000, seed=10)
#         print("nb lifetime data :", len(lifetime_data))
#         distri.fit_from_lifetime_data(lifetime_data)
#         print(distri.params)
#         assert np.allclose(distri.params, expected_params, rtol=1e-3)
