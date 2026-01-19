# pyright: basic
import numpy as np
import pytest
from pytest import approx

from relife.economic import RunToFailureReward
from relife.lifetime_model import EquilibriumDistribution
from relife.stochastic_process import RenewalProcess, RenewalRewardProcess


class TestDistribution:
    def test_renewal_density(self, distribution):
        renewal_process = RenewalProcess(distribution, first_lifetime_model=EquilibriumDistribution(distribution))
        timeline, renewal_density = renewal_process.renewal_density(100, 200)
        assert timeline.shape == (200,)
        assert renewal_density.shape == (200,)
        assert renewal_density[..., -1:] == approx(1 / distribution.mean(), rel=1e-4)

    def test_expected_total_reward(self, distribution):
        reward = RunToFailureReward(cf=1.0)
        renewal_reward_process = RenewalRewardProcess(distribution, reward)
        timeline_m, m = renewal_reward_process.renewal_function(100.0, 200)
        assert timeline_m.shape == m.shape == (200,)
        timeline_z, z = renewal_reward_process.expected_total_reward(100, 200)
        assert timeline_z.shape == z.shape == (200,)
        assert m == approx(z, rel=1e-4)

    def test_renewal_reward_process_vec(self, distribution):
        cf0 = 1
        n = 3
        cf = cf0 / n

        rrp0 = RenewalRewardProcess(
            distribution,
            RunToFailureReward(cf0),
            discounting_rate=0.04,
        )
        rrp = RenewalRewardProcess(
            distribution,
            RunToFailureReward(np.full((n, 1), cf)),
            discounting_rate=0.04,
        )

        timeline_z, z = rrp.expected_total_reward(100, 200)  # (3, nb_steps)
        assert timeline_z.shape == (200,)
        assert z.shape == (3, 200)
        timeline_z0, z0 = rrp0.expected_total_reward(100, 200)  # (nb_steps,)
        assert timeline_z0.shape == (200,)
        assert z0.shape == (200,)
        assert z.shape == (n, 200)
        assert z0 == approx(z.sum(axis=0), rel=1e-4)

    @pytest.mark.skip(reason="was not tested in v1 and we will work on it later")
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_sample_lifetime_data(self, distribution):
        expected_params = distribution.params.copy()
        q1 = distribution.ppf(0.25)
        q3 = distribution.ppf(0.75)
        success = 0
        n = 100
        renewal_process = RenewalProcess(distribution)
        for i in range(n):
            lifetime_data = renewal_process.generate_failure_data(10000, 10 * q3, t0=0)
            try:  #  for gamma and loglogistic essentially (convergence errors may occcur)
                distribution.fit(**lifetime_data)
            except RuntimeError:
                continue
            ic = distribution.fitting_results.IC
            # params found are within params IC
            print("expected params :", expected_params)
            print("IC95 :", ic)
            if np.all(expected_params.reshape(-1, 1) >= ic[:, [0]]) and np.all(
                expected_params.reshape(-1, 1) <= ic[:, [1]]
            ):
                success += 1
        print(success / n)
        assert success >= 0.95 * n


class TestAgeReplacementDistribution:
    def test_renewal_density(self, frozen_ar_distribution):
        renewal_process = RenewalProcess(
            frozen_ar_distribution, first_lifetime_model=EquilibriumDistribution(frozen_ar_distribution)
        )
        timeline, renewal_density = renewal_process.renewal_density(100, 200)
        assert timeline.shape == (200,)
        assert renewal_density.shape == (200,)
        assert renewal_density[..., -1:] == approx(1 / frozen_ar_distribution.mean(), rel=1e-4)

    def test_expected_total_reward(self, frozen_ar_distribution):
        reward = RunToFailureReward(cf=1.0)
        renewal_reward_process = RenewalRewardProcess(frozen_ar_distribution, reward)
        timeline_m, m = renewal_reward_process.renewal_function(100.0, 200)
        assert timeline_m.shape == (200,)
        assert m.shape == (200,)
        timeline_z, z = renewal_reward_process.expected_total_reward(100, 200)
        assert timeline_z.shape == (200,)
        assert z.shape == (200,)
        assert m == approx(z, rel=1e-4)

    def test_renewal_reward_process_vec(self, frozen_ar_distribution):
        cf0 = 1
        n = 3
        cf = cf0 / n

        rrp0 = RenewalRewardProcess(
            frozen_ar_distribution,
            RunToFailureReward(cf0),
            discounting_rate=0.04,
        )
        rrp = RenewalRewardProcess(
            frozen_ar_distribution,
            RunToFailureReward(np.full((n, 1), cf)),
            discounting_rate=0.04,
        )
        timeline_z, z = rrp.expected_total_reward(100, 200)  # (3, nb_steps)
        assert timeline_z.shape == (200,)
        assert z.shape == (3, 200)
        timeline_z0, z0 = rrp0.expected_total_reward(100, 200)  # (nb_steps,)
        assert timeline_z0.shape == (200,)
        assert z0.shape == (200,)
        assert z.shape == (n, 200)
        assert z0 == approx(z.sum(axis=0), rel=1e-4)


class TestRegression:
    def test_renewal_density(self, frozen_regression):
        renewal_process = RenewalProcess(
            frozen_regression, first_lifetime_model=EquilibriumDistribution(frozen_regression)
        )
        timeline, renewal_density = renewal_process.renewal_density(100, 200)
        assert timeline.shape == (200,)
        assert renewal_density.shape == (3, 200)
        assert renewal_density[..., -1:] == approx(1 / frozen_regression.mean(), rel=1e-4)

    def test_expected_total_reward(self, frozen_regression):
        reward = RunToFailureReward(cf=1.0)
        renewal_reward_process = RenewalRewardProcess(frozen_regression, reward)
        timeline_m, m = renewal_reward_process.renewal_function(100.0, 200)
        assert timeline_m.shape == (200,)
        assert m.shape == (3, 200)
        timeline_z, z = renewal_reward_process.expected_total_reward(100, 200)
        assert timeline_z.shape == (200,)
        assert z.shape == (3, 200)
        assert m == approx(z, rel=1e-4)

    def test_renewal_reward_process_vec(self, frozen_regression):
        cf0 = 1
        n = 3
        cf = cf0 / n

        rrp0 = RenewalRewardProcess(
            frozen_regression,
            RunToFailureReward(cf0),
            discounting_rate=0.04,
        )
        rrp = RenewalRewardProcess(
            frozen_regression,
            RunToFailureReward(np.full((n, 1), cf)),
            discounting_rate=0.04,
        )
        timeline_z, z = rrp.expected_total_reward(100, 200)  # (3, nb_steps)

        assert timeline_z.shape == (200,)
        assert z.shape == (3, 200)

        timeline_z0, z0 = rrp0.expected_total_reward(100, 200)  # (3, nb_steps,)

        assert timeline_z0.shape == (200,)
        assert z0.shape == (n, 200)
        assert z.shape == (n, 200)
        assert z0 == approx(n * z, rel=1e-4)


class TestAgeReplacementRegression:
    def test_renewal_density(self, frozen_ar_regression):
        renewal_process = RenewalProcess(
            frozen_ar_regression, first_lifetime_model=EquilibriumDistribution(frozen_ar_regression)
        )
        timeline, renewal_density = renewal_process.renewal_density(100, 200)
        assert timeline.shape == (200,)
        assert renewal_density.shape == (3, 200)
        assert renewal_density[..., -1:] == approx(1 / frozen_ar_regression.mean(), rel=1e-4)

    def test_expected_total_reward(self, frozen_ar_regression):
        reward = RunToFailureReward(cf=1.0)
        renewal_reward_process = RenewalRewardProcess(frozen_ar_regression, reward)
        timeline_m, m = renewal_reward_process.renewal_function(100.0, 200)
        assert timeline_m.shape == (200,)
        assert m.shape == (3, 200)
        timeline_z, z = renewal_reward_process.expected_total_reward(100, 200)
        assert timeline_z.shape == (200,)
        assert z.shape == (3, 200)
        assert m == approx(z, rel=1e-4)

    def test_renewal_reward_process_vec(self, frozen_ar_regression):
        cf0 = 1
        n = 3
        cf = cf0 / n

        rrp0 = RenewalRewardProcess(
            frozen_ar_regression,
            RunToFailureReward(cf0),
            discounting_rate=0.04,
        )
        rrp = RenewalRewardProcess(
            frozen_ar_regression,
            RunToFailureReward(np.full((n, 1), cf)),
            discounting_rate=0.04,
        )
        timeline_z, z = rrp.expected_total_reward(100, 200)  # (3, nb_steps)
        assert timeline_z.shape == (200,)
        assert z.shape == (3, 200)
        timeline_z0, z0 = rrp0.expected_total_reward(100, 200)  # (3, nb_steps,)
        assert timeline_z0.shape == (200,)
        assert z0.shape == (
            n,
            200,
        )
        assert z.shape == (n, 200)
        assert z0 == approx(n * z, rel=1e-4)
