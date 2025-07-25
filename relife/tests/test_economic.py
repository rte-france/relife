from relife.economic import cost, RunToFailureReward, AgeReplacementReward


def test_cost(cf, cp, expected_out_shape):
    cost_arr = cost(cf = cf, cp = cp)
    assert cost_arr["cf"].shape == expected_out_shape(cf=cf, cp=cp)


def test_run_to_failure_reward(time, cf, expected_out_shape):
    reward = RunToFailureReward(cf)
    assert reward.conditional_expectation(time).shape == expected_out_shape(time=time, cf=cf)
    assert reward.sample(time).shape == expected_out_shape(time=time, cf=cf)

def test_age_replacement_reward(time, cf, cp, ar, expected_out_shape):
    reward = AgeReplacementReward(cf, cp, ar)
    assert reward.conditional_expectation(time).shape == expected_out_shape(time=time, cf=cf, cp=cp, ar=ar)
    assert reward.sample(time).shape == expected_out_shape(time=time, cf=cf, cp=cp, ar=ar)