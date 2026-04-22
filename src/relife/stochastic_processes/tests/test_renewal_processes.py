# pyright: basic
import numpy as np
import pytest
from pytest import approx

from relife.lifetime_models import EquilibriumDistribution, LeftTruncatedModel
from relife.lifetime_models._conditional_models import AgeReplacementModel
from relife.rewards import RunToFailureReward
from relife.stochastic_processes import RenewalProcess, RenewalRewardProcess
from relife.stochastic_processes._sample._iterables import RenewalProcessIterable
from relife.utils import get_nb_assets

from .utils import select_from_struct


class TestDistribution:
    def test_renewal_density(self, distribution):
        renewal_process = RenewalProcess(
            distribution, first_lifetime_model=EquilibriumDistribution(distribution)
        )
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
        q3 = distribution.ppf(0.75)
        success = 0
        n = 100
        renewal_process = RenewalProcess(distribution)
        for i in range(n):
            lifetime_data = renewal_process.generate_failure_data(
                10000, 10 * q3, t0=0, seed=21
            )
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
            frozen_ar_distribution,
            first_lifetime_model=EquilibriumDistribution(frozen_ar_distribution),
        )
        timeline, renewal_density = renewal_process.renewal_density(100, 200)
        assert timeline.shape == (200,)
        assert renewal_density.shape == (200,)
        assert renewal_density[..., -1:] == approx(
            1 / frozen_ar_distribution.mean(), rel=1e-4
        )

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
            frozen_regression,
            first_lifetime_model=EquilibriumDistribution(frozen_regression),
        )
        timeline, renewal_density = renewal_process.renewal_density(100, 200)
        assert timeline.shape == (200,)
        assert renewal_density.shape == (3, 200)
        assert renewal_density[..., -1:] == approx(
            1 / frozen_regression.mean(), rel=1e-4
        )

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
            frozen_ar_regression,
            first_lifetime_model=EquilibriumDistribution(frozen_ar_regression),
        )
        timeline, renewal_density = renewal_process.renewal_density(100, 200)
        assert timeline.shape == (200,)
        assert renewal_density.shape == (3, 200)
        assert renewal_density[..., -1:] == approx(
            1 / frozen_ar_regression.mean(), rel=1e-4
        )

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


def test_age_replacement_sampling(distribution, ar):
    renewal_process = RenewalProcess(
        distribution,
        first_lifetime_model=EquilibriumDistribution(distribution),
    )

    trial_model = AgeReplacementModel(distribution).freeze(ar)
    nb_assets = get_nb_assets(*trial_model.args)
    ar_reshaped = trial_model.args[0]

    nb_samples = 10
    t0 = distribution.ppf(0.25)
    tf = 10 * distribution.ppf(0.75)

    iterable = RenewalProcessIterable(
        renewal_process, nb_samples, (t0, tf), ar=ar, seed=21
    )
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # Check that all times are less than ar for each asset
    for i in range(nb_assets):
        ar_asset = np.atleast_1d(ar_reshaped)[i]
        select_asset = select_from_struct(struct_array, asset_id=i)
        assert (select_asset["time"] <= ar_asset + 1e-5).all()


def test_left_truncated_sampling(distribution, a0):

    renewal_process = RenewalProcess(
        distribution,
        first_lifetime_model=EquilibriumDistribution(distribution),
    )

    trial_model = LeftTruncatedModel(distribution).freeze(a0)
    a0_reshaped = trial_model.args[0]

    tf = 10 * distribution.ppf(0.75)
    nb_samples = 100

    iterable = RenewalProcessIterable(
        renewal_process, nb_samples, (0, tf), a0=a0, seed=21
    )
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # check first entries are a0 for each sample
    for i in range(nb_samples):
        select_sample = select_from_struct(struct_array, sample_id=i)
        first_entries = select_sample["entry"][select_sample["entry"] > 0].reshape(
            a0_reshaped.shape
        )
        np.testing.assert_equal(first_entries, a0_reshaped)


def test_age_replacement_regression_sampling(frozen_regression, ar):
    renewal_process = RenewalProcess(
        frozen_regression,
        first_lifetime_model=EquilibriumDistribution(frozen_regression),
    )

    trial_model = AgeReplacementModel(frozen_regression).freeze(ar)
    nb_assets = get_nb_assets(*trial_model.args)
    ar_reshaped = trial_model.args[0]

    t0 = frozen_regression.ppf(0.25).min()
    tf = 10 * frozen_regression.ppf(0.75).max()
    nb_samples = 10

    iterable = RenewalProcessIterable(
        renewal_process, nb_samples, (t0, tf), ar=ar, seed=21
    )
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("asset_id", "sample_id", "timeline"))

    # check all times are bounded by the age of replacement
    # add a small constant for numerical approximations
    for i in range(nb_assets):
        ar_asset = np.atleast_1d(ar_reshaped)[i]
        times = select_from_struct(struct_array, asset_id=i)["time"]
        assert (times <= ar_asset + 1e-5).all()
