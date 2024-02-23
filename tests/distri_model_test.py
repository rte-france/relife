import numpy as np
import pytest

from relife2.data import load_power_transformer
from relife2.survival import exponential


@pytest.fixture(scope="module")
def databook():
    return load_power_transformer()


@pytest.fixture(scope="module")
def exponential_distri(databook):
    return exponential(databook)


@pytest.fixture(scope="module")
def weibull_distri(databook):
    pass


@pytest.fixture(scope="module")
def gompertz_distri(databook):
    pass


@pytest.fixture(scope="module")
def gamma_distri(databook):
    pass


@pytest.fixture(scope="module")
def loglogistic_distri(databook):
    pass


@pytest.mark.parametrize(
    "model",
    "params",
    [
        (exponential_distri, 0.00795203),
        (weibull_distri, [3.46597395, 0.01227849]),
        (gompertz_distri, [0.00865741, 0.06062632]),
        (gamma_distri, [5.3571091, 0.06622822]),
        (loglogistic_distri, [3.92614064, 0.0133325]),
    ],
)
def test_sf(model, params):
    assert model.sf(model.median(params)) == pytest.approx(0.5, rel=1e-3)


@pytest.mark.parametrize(
    "model",
    "params",
    [
        (exponential_distri, 0.00795203),
        (weibull_distri, [3.46597395, 0.01227849]),
        (gompertz_distri, [0.00865741, 0.06062632]),
        (gamma_distri, [5.3571091, 0.06622822]),
        (loglogistic_distri, [3.92614064, 0.0133325]),
    ],
)
def test_rvs(model, params):
    size = 10
    assert model.rvs(params, size=size).shape == (size,)


# def test_mean(model):
#     assert super(type(model), model).mean() == pytest.approx(
#         model.mean(), rel=1e-3
#     )


@pytest.mark.parametrize(
    "model",
    "params",
    [
        (exponential_distri, 0.00795203),
        (weibull_distri, [3.46597395, 0.01227849]),
        (gompertz_distri, [0.00865741, 0.06062632]),
        (gamma_distri, [5.3571091, 0.06622822]),
        (loglogistic_distri, [3.92614064, 0.0133325]),
    ],
)
def test_mrl(model, params):
    t = np.arange(10)
    assert model.mrl(params, t).shape == (t.size,)


@pytest.mark.parametrize(
    "model",
    "params",
    [
        (exponential_distri, 0.00795203),
        (weibull_distri, [3.46597395, 0.01227849]),
        (gompertz_distri, [0.00865741, 0.06062632]),
        (gamma_distri, [5.3571091, 0.06622822]),
        (loglogistic_distri, [3.92614064, 0.0133325]),
    ],
)
def test_fit(model, params):
    model.fit()
    assert model.fitting_params == pytest.approx(params, rel=1e-3)


# def test_minimum_distribution(model, data):
#     params = model.params.copy()
#     n = np.ones((data.size, 1))
#     model = MinimumDistribution(model).fit(*data.astuple(), args=(n,))
#     assert model.params == pytest.approx(params, rel=1e-3)
