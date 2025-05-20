import pytest
import numpy as np

from relife.policy import OneCycleAgeReplacementPolicy, AgeReplacementPolicy


def test_asymptotic_expected_equivalent_annual_cost(run_to_failure_policy):
    qa = run_to_failure_policy.asymptotic_expected_equivalent_annual_cost()
    timeline, q = run_to_failure_policy.expected_equivalent_annual_cost(400, nb_steps=2000)
    assert timeline.shape == q.shape == (5, 2000)
    assert q[..., -1].flatten() == pytest.approx(qa.flatten(), rel=1e-1)


@pytest.mark.skip(reason="conflict between ar and period_before_discounting for ls_integrate computation")
def test_one_cycle_(distribution, cf, cp, discounting_rate):
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
    # TODO : error in ReLife 1 too
    # policy.asymptotic_expected_equivalent_annual_cost() computes ls_integrate where b < a
    # a = period_before_discounting (0.1)
    # because b = np.minimum(ar, b), with b = inf BUT some ar < 0.1
    assert np.all(
        policy1.asymptotic_expected_equivalent_annual_cost()
        > policy.asymptotic_expected_equivalent_annual_cost()
    ) and np.all(
        policy0.asymptotic_expected_equivalent_annual_cost()
        > policy.asymptotic_expected_equivalent_annual_cost()
    )


@pytest.mark.skip(reason="conflict between ar and period_before_discounting for ls_integrate computation")
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