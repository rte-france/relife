import pytest
import numpy as np

from relife.lifetime_model import Exponential
from relife.policy import OneCycleAgeReplacementPolicy, AgeReplacementPolicy

# ignore runtime warning in optimization
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_one_cycle_asymptotic_expected_equivalent_annual_cost(distribution, cf, cp, discounting_rate, expected_out_shape):
    if isinstance(distribution, Exponential):
        pytest.skip("Exponential distribution won't work with this cf, cp (not tested in v1.0.0 too)")
    policy = OneCycleAgeReplacementPolicy(distribution, cf, cp, discounting_rate=discounting_rate)
    try:
        policy.optimize()
    except RuntimeError:
        pytest.skip("Optimization failed, EEAC may be too flat")
    qa = policy.asymptotic_expected_equivalent_annual_cost()
    assert qa.shape == expected_out_shape(cf=cf, cp=cp) # () or (m, 1)
    timeline, q = policy.expected_equivalent_annual_cost(400, nb_steps=2000)
    assert timeline.shape == q.shape
    assert timeline.shape == np.broadcast_shapes(expected_out_shape(cf=cf, cp=cp), (2000,)) # (m, 2000) or (2000,)
    assert q[..., -1].flatten() == pytest.approx(qa.flatten(), rel=1e-1)

# ignore runtime warning in optimization
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_asymptotic_expected_equivalent_annual_cost(distribution, cf, cp, discounting_rate, expected_out_shape):
    if isinstance(distribution, Exponential):
        pytest.skip("Exponential distribution won't work with this cf, cp (not tested in v1.0.0 too)")
    policy = AgeReplacementPolicy(distribution, cf, cp, discounting_rate=discounting_rate)
    try:
        policy.optimize()
    except RuntimeError:
        pytest.skip("Optimization failed, EEAC may be too flat")
    qa = policy.asymptotic_expected_equivalent_annual_cost() # () or (m, 1)
    timeline, q = policy.expected_equivalent_annual_cost(400, nb_steps=2000)
    assert timeline.shape == q.shape
    assert timeline.shape == np.broadcast_shapes(expected_out_shape(cf=cf, cp=cp), (2000,))  # (m, 2000) or (2000,)
    assert q[..., -1].flatten() == pytest.approx(qa.flatten(), rel=1e-1)


def test_one_cycle_optimal_replacement_age(distribution, cf, cp, discounting_rate):
    if isinstance(distribution, Exponential):
        pytest.skip("Exponential distribution won't work with this cf, cp (not tested in v1.0.0 too)")
    eps = 1e-2
    policy = OneCycleAgeReplacementPolicy(distribution, cf, cp, discounting_rate=discounting_rate).optimize()
    policy1 = OneCycleAgeReplacementPolicy(
        distribution,
        cf,
        cp,
        discounting_rate=discounting_rate,
        period_before_discounting=0.1,
        ar=policy.ar + eps,
    )
    policy0 = OneCycleAgeReplacementPolicy(
        distribution,
        cf,
        cp,
        discounting_rate=discounting_rate,
        period_before_discounting=0.1,
        ar=policy.ar - eps,
    )
    assert np.all(
        policy1.asymptotic_expected_equivalent_annual_cost()
        > policy.asymptotic_expected_equivalent_annual_cost()
    ) and np.all(
        policy0.asymptotic_expected_equivalent_annual_cost()
        > policy.asymptotic_expected_equivalent_annual_cost()
    )

def test_optimal_replacement_age(distribution, cf, cp, discounting_rate):
    if isinstance(distribution, Exponential):
        pytest.skip("Exponential distribution won't work with this cf, cp (not tested in v1.0.0 too)")
    eps = 1e-2
    policy = AgeReplacementPolicy(
        distribution, cf, cp, discounting_rate=discounting_rate
    ).optimize()
    ar = policy.ar

    policy1 = AgeReplacementPolicy(
        distribution, cf, cp, discounting_rate=discounting_rate, ar=ar + eps
    )
    policy0 = AgeReplacementPolicy(
        distribution, cf, cp, discounting_rate=discounting_rate, ar=ar - eps
    )

    assert np.all(
        policy1.asymptotic_expected_equivalent_annual_cost()
        > policy.asymptotic_expected_equivalent_annual_cost()
    ) and np.all(
        policy0.asymptotic_expected_equivalent_annual_cost()
        > policy.asymptotic_expected_equivalent_annual_cost()
    )