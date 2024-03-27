import numpy as np
import pytest

from relife2.survival import Exponential, Gamma, Gompertz, Weibull
from relife2.survival.data import load_power_transformer


@pytest.fixture(scope="module")
def data():
    return load_power_transformer()


@pytest.fixture(
    scope="module",
    params=[
        Exponential(0.00795203),
        Weibull(3.46597395, 0.01227849),
        Gompertz(0.00865741, 0.06062632),
        Gamma(5.3571091, 0.06622822),
        # logLogistic(3.92614064, 0.0133325),
    ],
)
def model(request):
    return request.param


def test_sf(model):
    assert model.sf(model.median()) == pytest.approx(0.5, rel=1e-3)


def test_rvs(model):
    size = 10
    assert model.rvs(size=size).shape == (size,)


# /!\ depends upon LS_INTEGRATE
def test_mean(model):
    assert super(type(model), model).mean() == pytest.approx(
        model.mean(), rel=1e-3
    )


def test_mrl(model):
    t = np.arange(10)
    assert model.mrl(t).shape == (t.size,)


def test_fit(model, data):
    params = model.params.values.copy()
    model.fit(
        data[0, :],
        complete_indicators=data[1, :] == 1,
        right_censored_indicators=data[1, :] == 0,
        entry=data[2, :],
    )
    assert model.fitting_params.values == pytest.approx(params, rel=1e-3)


# def test_minimum_distribution(model, data):
#     params = model.params.copy()
#     n = np.ones((data.size, 1))
#     model = MinimumDistribution(model).fit(*data.astuple(), args=(n,))
#     assert model.params == pytest.approx(params, rel=1e-3)
