import pytest
import numpy as np

from relife.policy import OneCycleAgeReplacementPolicy, AgeReplacementPolicy


def test_asymptotic_expected_equivalent_annual_cost(age_replacement_policy):
    age_replacement_policy.optimize()
    qa = age_replacement_policy.asymptotic_expected_equivalent_annual_cost()
    timeline, q = age_replacement_policy.expected_equivalent_annual_cost(400, nb_steps=2000)
    assert timeline.shape == q.shape == (5, 2000)
    assert q[..., -1].flatten() == pytest.approx(qa.flatten(), rel=1e-1)


def test_one_cycle(distribution, cf, cp, discounting_rate):
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