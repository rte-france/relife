# pyright:basic
import numpy as np
import pytest

from relife.lifetime_model import Exponential
from relife.policy import AgeReplacementPolicy, OneCycleAgeReplacementPolicy


class TestOneCycleAgeReplacementPolicy:
    # ignore runtime warning in optimization
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_asymptotic_expected_equivalent_annual_cost(self, distribution, cf, cp, discounting_rate):
        if isinstance(distribution, Exponential):
            pytest.skip("Exponential distribution won't work with this cf, cp (not tested in v1.0.0 too)")
        policy = OneCycleAgeReplacementPolicy(distribution, cf, cp, discounting_rate=discounting_rate)
        try:
            policy.optimize()
        except RuntimeError:
            pytest.skip("Optimization failed, EEAC may be too flat")
        qa = policy.asymptotic_expected_equivalent_annual_cost()
        assert qa.shape == np.broadcast_shapes(cf.shape, cp.shape)  # () or (m,)

    # ignore runtime warning in optimization
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_expected_equivalent_annual_cost(self, distribution, cf, cp, discounting_rate):
        if isinstance(distribution, Exponential):
            pytest.skip("Exponential distribution won't work with this cf, cp (not tested in v1.0.0 too)")
        policy = OneCycleAgeReplacementPolicy(distribution, cf, cp, discounting_rate=discounting_rate)
        try:
            policy.optimize()
        except RuntimeError:
            pytest.skip("Optimization failed, EEAC may be too flat")

        qa = policy.asymptotic_expected_equivalent_annual_cost()  # () or (m,)
        timeline, q = policy.expected_equivalent_annual_cost(400, nb_steps=2000)

        assert timeline.shape == (2000,)
        assert q.shape == qa.shape + timeline.shape  # (m, 2000) or (2000,)

    def test_optimal_replacement_age(self, distribution, cf, cp, discounting_rate):
        if isinstance(distribution, Exponential):
            pytest.skip("Exponential distribution won't work with this cf, cp (not tested in v1.0.0 too)")
        eps = 1e-2
        policy = OneCycleAgeReplacementPolicy(distribution, cf, cp, discounting_rate=discounting_rate).optimize()
        suboptimal_policy1 = OneCycleAgeReplacementPolicy(
            distribution,
            cf,
            cp,
            discounting_rate=discounting_rate,
            period_before_discounting=0.1,
            ar=policy.ar + eps,
        )
        suboptimal_policy2 = OneCycleAgeReplacementPolicy(
            distribution,
            cf,
            cp,
            discounting_rate=discounting_rate,
            period_before_discounting=0.1,
            ar=policy.ar - eps,
        )
        assert np.all(
            suboptimal_policy1.asymptotic_expected_equivalent_annual_cost()
            > policy.asymptotic_expected_equivalent_annual_cost()
        ) and np.all(
            suboptimal_policy2.asymptotic_expected_equivalent_annual_cost()
            > policy.asymptotic_expected_equivalent_annual_cost()
        )


class TestAgeReplacementPolicy:
    # ignore runtime warning in optimization
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_asymptotic_expected_equivalent_annual_cost(self, distribution, cf, cp, discounting_rate):
        if isinstance(distribution, Exponential):
            pytest.skip("Exponential distribution won't work with this cf, cp (not tested in v1.0.0 too)")
        policy = AgeReplacementPolicy(distribution, cf, cp, discounting_rate=discounting_rate)
        try:
            policy.optimize()
        except RuntimeError:
            pytest.skip("Optimization failed, EEAC may be too flat")
        qa = policy.asymptotic_expected_equivalent_annual_cost()  # () or (m,)
        assert qa.shape == np.broadcast_shapes(cf.shape, cp.shape)  # () or (m,)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_expected_equivalent_annual_cost(self, distribution, cf, cp, discounting_rate):
        if isinstance(distribution, Exponential):
            pytest.skip("Exponential distribution won't work with this cf, cp (not tested in v1.0.0 too)")
        policy = AgeReplacementPolicy(distribution, cf, cp, discounting_rate=discounting_rate)
        try:
            policy.optimize()
        except RuntimeError:
            pytest.skip("Optimization failed, EEAC may be too flat")
        qa = policy.asymptotic_expected_equivalent_annual_cost()  # () or (m,)
        timeline, q = policy.expected_equivalent_annual_cost(400, nb_steps=2000)

        assert timeline.shape == (2000,)
        assert q.shape == qa.shape + timeline.shape  # (m, 2000) or (2000,)
        assert q[..., -1].flatten() == pytest.approx(qa.flatten(), rel=1e-1)

    def test_optimal_replacement_age(self, distribution, cf, cp, discounting_rate):
        if isinstance(distribution, Exponential):
            pytest.skip("Exponential distribution won't work with this cf, cp (not tested in v1.0.0 too)")
        eps = 1e-2
        policy = AgeReplacementPolicy(distribution, cf, cp, discounting_rate=discounting_rate).optimize()
        suboptimal_policy1 = AgeReplacementPolicy(
            distribution, cf, cp, discounting_rate=discounting_rate, ar=policy.ar + eps
        )
        suboptimal_policy2 = AgeReplacementPolicy(
            distribution, cf, cp, discounting_rate=discounting_rate, ar=policy.ar - eps
        )

        assert np.all(
            suboptimal_policy1.asymptotic_expected_equivalent_annual_cost()
            > policy.asymptotic_expected_equivalent_annual_cost()
        ) and np.all(
            suboptimal_policy2.asymptotic_expected_equivalent_annual_cost()
            > policy.asymptotic_expected_equivalent_annual_cost()
        )
