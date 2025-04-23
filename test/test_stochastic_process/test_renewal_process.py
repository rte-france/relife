import numpy as np
from pytest import approx

from relife.lifetime_model import EquilibriumDistribution, AgeReplacementModel
from relife.stochastic_process import RenewalProcess


# @pytest.fixture(
#     scope="module",
#     params=[
#         Weibull(2, 0.05),
#         Gompertz(0.01, 0.1),
#         Gamma(2, 0.05),
#         LogLogistic(3, 0.05),
#     ],
# )
# def distribution(request):
#     return request.param
#
#
# @pytest.fixture(scope="module", params=[None, AFT, ProportionalHazard])
# def regression(request):
#     return request.param
#
#
# @pytest.fixture(
#     scope="module",
#     params=[
#         None,
#         AgeReplacementModel,
#     ],
# )
# def age_replacement_model(request):
#     return request.param
#
#
# @pytest.fixture(scope="module")
# def frozen_model(distribution, regression, age_replacement_model):
#     model = distribution
#     args = ()
#     if regression is not None:
#         covar = np.arange(0.0, 0.6, 0.1).reshape(-1, 2)
#         model = regression(model, coef=(np.log(2), np.log(2)))
#         args = (covar,) + args
#     if age_replacement_model is not None:
#         tmax = model.isf(0.75, *args)
#         model = age_replacement_model(model)
#         args = (tmax,) + args
#     return model.freeze(*args)
#

# test functions
def test_renewal_process_distribution(distribution):
    timeline = np.arange(0, 100, 0.5)
    renewal_process = RenewalProcess(distribution, model1=EquilibriumDistribution(distribution))
    assert renewal_process.renewal_density(timeline)[..., -1:] == approx(1 / distribution.mean(), rel=1e-4)

    ar_distribution = AgeReplacementModel(distribution)
    ar = distribution.isf(0.75)

    renewal_process = RenewalProcess(ar_distribution.freeze(ar), model1=EquilibriumDistribution(ar_distribution).freeze(ar))
    assert renewal_process.renewal_density(timeline)[..., -1:] == approx(1 / distribution.mean(), rel=1e-4)


def test_renewal_process_regression(regression):
    pass



#
#
#
#
#
#     model1 =
#     rp = RenewalProcess(
#         frozen_model,
#         model1=EquilibriumDistribution(frozen_model.baseline),
#         model_args=model_args,
#         model1_args=model_args,
#         nb_assets=nb_assets,
#     )
#     y0 =
#     y = rp.renewal_density(t)
#     assert y[..., -1:] == pytest.approx(y0, rel=1e-4)
#
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
