import pytest
import numpy as np

from relife.economic import cost, RunToFailureReward, AgeReplacementReward

M = 10
N = 3


@pytest.fixture(
    params=[
        np.float64(1),
        np.ones((1,), dtype=np.float64),
        np.ones((N,), dtype=np.float64),
        np.ones((1, 1), dtype=np.float64),
        np.ones((M, 1), dtype=np.float64),
        np.ones((1, N), dtype=np.float64),
        np.ones((M, 3), dtype=np.float64),
    ],
    ids=lambda time: f"time:{time.shape}",
)
def time(request):
    return request.param


@pytest.fixture(
    params=[
        np.ones((), dtype=np.float64),
        np.ones((1,), dtype=np.float64),
        np.ones((M,), dtype=np.float64),
        np.ones((M, 1), dtype=np.float64),
    ],
    ids=lambda ar: f"ar:{ar.shape}",
)
def ar(request):
    return request.param


@pytest.fixture(
    params=[
        np.ones((), dtype=np.float64) * 3.,
        np.ones((1,), dtype=np.float64) * 3.,
        np.ones((M,), dtype=np.float64) * 3.,
        np.ones((M, 1), dtype=np.float64) * 3.,
    ],
    ids=lambda cf: f"cf:{cf.shape}",
)
def cf(request):
    return request.param


@pytest.fixture(
    params=[
        np.ones((), dtype=np.float64) * 3.,
        np.ones((1,), dtype=np.float64) * 3.,
        np.ones((M,), dtype=np.float64) * 3.,
        np.ones((M, 1), dtype=np.float64) * 3.,
    ],
    ids=lambda cp: f"cp:{cp.shape}",
)
def cp(request):
    return request.param


@pytest.fixture
def expected_out_shape():
    def _expected_out_shape(**kwargs):
        def shape_contrib(**kwargs):
            yield ()  # yield at least (), in case kwargs is empty
            for k, v in kwargs.items():
                match k:
                    case "cf" | "cp" | "ar" if v.ndim == 2 or v.ndim == 0:
                        yield v.shape
                    case "cf" | "cp" | "ar" if v.ndim == 1:
                        yield v.size, 1
                    case _:
                        yield v.shape

        return np.broadcast_shapes(*tuple(shape_contrib(**kwargs)))

    return _expected_out_shape


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