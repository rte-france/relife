import numpy as np
import pytest
from pytest import approx

from relife.economic import RunToFailureReward
from relife.lifetime_model import EquilibriumDistribution
from relife.lifetime_model.conditional_model import LeftTruncatedModel
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
            lifetime_data = renewal_process.generate_failure_data(10000, (0.0, 10 * q3))
            try:  #  for gamma and loglogistic essentially (convergence errors may occcur)
                distribution.fit(lifetime_data["time"], lifetime_data["event"], lifetime_data["entry"])
            except RuntimeError:
                continue
            ic = distribution.fitting_results.IC
            # params found are within params IC
            print(f"i: {i}")
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

    def test_sampling(self, frozen_ar_distribution):
        renewal_process = RenewalProcess(
            frozen_ar_distribution, first_lifetime_model=EquilibriumDistribution(frozen_ar_distribution)
        )
        ar = frozen_ar_distribution.args[0]
        n_assets = 1 if isinstance(ar, float) else ar.shape[0]
        n_samples = 10
        t0 = frozen_ar_distribution.ppf(0.25)
        tf = 10 * frozen_ar_distribution.ppf(0.75)
        sample = renewal_process.sample(n_samples, (t0, tf))

        # Check that all times are less than ar for each asset
        for i in range(n_assets):
            ar_asset = ar if isinstance(ar, float) else ar[i]
            select_asset = sample._select_from_struct(asset_id=i)
            np.testing.assert_array_less(select_asset["time"], ar_asset + 1e-5)


class TestLeftTruncatedDistribution:
    def test_sampling(self, distribution, a0):
        first_lifetime_model = LeftTruncatedModel(distribution).freeze(a0)
        renewal_process = RenewalProcess(distribution, first_lifetime_model=first_lifetime_model)
        tf = 10 * distribution.ppf(0.75)
        sample = renewal_process.sample(100, (0.0, tf))

        # check first entries are a0 for each sample
        for i in range(100):
            select_sample = sample._select_from_struct(sample_id=i)
            first_entries = select_sample["entry"][select_sample["entry"] > 0].reshape(a0.shape)
            np.testing.assert_equal(first_entries, a0)


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

    def test_sampling(self, frozen_ar_regression):
        renewal_process = RenewalProcess(
            frozen_ar_regression, first_lifetime_model=EquilibriumDistribution(frozen_ar_regression)
        )
        t0 = frozen_ar_regression.ppf(0.25).min()
        tf = 10 * frozen_ar_regression.ppf(0.75).max()
        n_samples = 10
        sample = renewal_process.sample(n_samples, (t0, tf))

        # check all times are bounded by the age of replacement
        # add a small constant for numerical approximations
        for i in range(frozen_ar_regression.args[0].shape[0]):
            times = sample._select_from_struct(asset_id=i)["time"]
            np.testing.assert_array_less(times, frozen_ar_regression.args[0][i].item() + 1e-5)
