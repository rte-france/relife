# pyright: basic

from relife.economic import AgeReplacementReward, RunToFailureReward


def test_run_to_failure_reward(time, cf, expected_out_shape):
    reward = RunToFailureReward(cf)
    assert reward.conditional_expectation(time).shape == expected_out_shape(time=time, cf=cf)
    assert reward.sample(time).shape == expected_out_shape(time=time, cf=cf)


def test_age_replacement_reward(time, cf, cp, ar, expected_out_shape):
    reward = AgeReplacementReward(cf, cp, ar)
    assert reward.conditional_expectation(time).shape == expected_out_shape(time=time, cf=cf, cp=cp, ar=ar)
    assert reward.sample(time).shape == expected_out_shape(time=time, cf=cf, cp=cp, ar=ar)
