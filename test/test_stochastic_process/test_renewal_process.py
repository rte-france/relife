from pytest import approx

from relife.lifetime_model import EquilibriumDistribution
from relife.stochastic_process import RenewalProcess


class TestDistribution:
    def test_renewal_density(self, distribution):
        renewal_process = RenewalProcess(distribution, model1=EquilibriumDistribution(distribution))
        assert renewal_process.renewal_density(100, 200).shape == (200,)
        assert renewal_process.renewal_density(100, 200)[..., -1:] == approx(1 / distribution.mean(), rel=1e-4)


class TestRegression:
    def test_renewal_density(self, frozen_regression):
        renewal_process = RenewalProcess(frozen_regression, model1=EquilibriumDistribution(frozen_regression))
        assert renewal_process.renewal_density(100, 200).shape == (3, 200)
        assert renewal_process.renewal_density(100, 200)[..., -1:] == approx(1 / frozen_regression.mean(), rel=1e-4)


class TestAgeReplacementDistribution:
    def test_renewal_density(self, frozen_ar_distribution):
        renewal_process = RenewalProcess(frozen_ar_distribution, model1=EquilibriumDistribution(frozen_ar_distribution))
        assert renewal_process.renewal_density(100, 200).shape == (3, 200)
        assert renewal_process.renewal_density(100, 200)[..., -1:] == approx(
            1 / frozen_ar_distribution.mean(), rel=1e-4
        )


class TestAgeReplacementRegression:
    def test_renewal_density(self, frozen_ar_regression):
        renewal_process = RenewalProcess(frozen_ar_regression, model1=EquilibriumDistribution(frozen_ar_regression))
        assert renewal_process.renewal_density(100, 200).shape == (3, 200)
        assert renewal_process.renewal_density(100, 200)[..., -1:] == approx(1 / frozen_ar_regression.mean(), rel=1e-4)


# def test_renewal_reward_process(distribution, regression):
#     renewal_reward_process = RenewalRewardProcess(distribution, reward(cf=1))


# @pytest.mark.skip(reason="no way of currently testing this")
# def test_renewal_reward_process(model_args_nb_assets):
#     t = np.arange(0, 100, 0.5)
#     model, model_args, nb_assets = model_args_nb_assets
#     reward = run_to_failure_rewards(cf=1)
#     rrp = RenewalRewardProcess(
#         model,
#         reward,
#         model_args=model_args,
#         nb_assets=nb_assets,
#     )
#     m = rrp.renewal_function(t)
#     z = rrp.expected_total_reward(t)
#     assert m == pytest.approx(z, rel=1e-4)
#
# @pytest.mark.skip(reason="no way of currently testing this")
# def test_renewal_reward_process_vec(model_args_nb_assets):
#     t = np.arange(0, 100, 0.5)
#     cf0 = 1
#     discounting_rate = 0.04
#     model, model_args, nb_assets = model_args_nb_assets
#
#     nb_assets = max(
#         tuple(map(lambda x: x.shape[0] if x.ndim >= 1 else 1, model_args)), default=1
#     )
#     n = (
#         5 if nb_assets == 1 else nb_assets
#     )  # vectorizes in 5 assets else equals the number of assets (could be a float too)
#     cf = cf0 / n
#
#     rrp0 = RenewalRewardProcess(
#         model,
#         run_to_failure_rewards(cf=cf0),
#         model_args=model_args,
#         discounting_rate=discounting_rate,
#         nb_assets=nb_assets,
#     )
#     rrp = RenewalRewardProcess(
#         model,
#         run_to_failure_rewards(cf=np.full((n, 1), cf)),
#         model_args=model_args,
#         discounting_rate=discounting_rate,
#         nb_assets=n,
#     )
#     z0 = rrp0.expected_total_reward(t)
#     z = rrp.expected_total_reward(t)
#     # if one asset, then z has 2 dim with n lines of expected_total_reward
#     if nb_assets == 1:
#         assert z0 == pytest.approx(z.sum(axis=0), rel=1e-4)
#     # if assets, then z0 has already nb_assets lines of expected_total_reward on first dim
#     else:
#         assert z0 == pytest.approx(n * z, rel=1e-4)
