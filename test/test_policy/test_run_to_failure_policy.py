import numpy as np
import pytest

from relife.policy import OneCycleRunToFailurePolicy, RunToFailurePolicy


def test_one_cycle_asymptotic_expected_equivalent_annual_cost(distribution, cf, discounting_rate, expected_out_shape):
    policy = OneCycleRunToFailurePolicy(distribution, cf, discounting_rate=discounting_rate)
    qa = policy.asymptotic_expected_equivalent_annual_cost() # () or (m, 1)
    timeline, q = policy.expected_equivalent_annual_cost(400, nb_steps=2000)
    assert timeline.shape == q.shape
    assert timeline.shape == q.shape == np.broadcast_shapes(expected_out_shape(cf=cf), (2000,)) # (m, 2000) or (2000,)
    assert q[..., -1].flatten() == pytest.approx(qa.flatten(), rel=1e-1)


def test_asymptotic_expected_equivalent_annual_cost(distribution, cf, discounting_rate, expected_out_shape):
    policy = RunToFailurePolicy(distribution, cf, discounting_rate=discounting_rate)
    qa = policy.asymptotic_expected_equivalent_annual_cost()
    timeline, q = policy.expected_equivalent_annual_cost(400, nb_steps=2000)
    assert timeline.shape == q.shape
    assert timeline.shape == np.broadcast_shapes(expected_out_shape(cf=cf), (2000,)) # (m, 2000) or (2000,)
    assert q[..., -1].flatten() == pytest.approx(qa.flatten(), rel=1e-1)


# FIXME : does not work because now max ndim in ls_integrate is 2d, here it's 3d -> broadcasting error
# possible solutions :
# 1. skip this test (obsolete) (-> recommanded)
# 2. set ndim (ReLife 1)
# 3. create a more complex mecanism to infer ndim inside ls_integrate (future improvements?)
# def test_expected_total_cost_vec(policy_vec):
#     batch_size = 3
#     timeline = np.arange(0, 100, 0.5)
#     z = policy_vec.expected_total_cost(timeline)
#     assert z.sum(axis=0) == pytest.approx(batch_size * z[0, ...], rel=1e-4)


# FIXME : does not work for end_time == 0.
#  In ReLife 1, when end_time == 0., atleast one sample was returned.
# possible solutions :
# 1. skip this test (obsolete) (-> recommanded), return one sample with end_time 0. has no sense
# 2.  change still_valid update order in sample_routine ?
# def test_sample(policy):
#     nb_assets = 5  # supposed to be set at initialization
#     nb_samples = 10
#     if isinstance(policy, RunToFailure):
#         data = policy.sample(nb_samples, 0.0)
#     else:
#         data = policy.sample(nb_samples)
#     assert len(data) == nb_samples * nb_assets
