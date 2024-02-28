import numpy as np
import pytest

from relife2.data import load_power_transformer
from relife2.survival.parametric import exponential


@pytest.fixture(scope="module")
def databook():
    return load_power_transformer()


@pytest.fixture(scope="module")
def exponential_distri(databook):
    return exponential(databook)


# @pytest.fixture(scope="module")
# def weibull_distri(databook):
#     pass


# @pytest.fixture(scope="module")
# def gompertz_distri(databook):
#     pass


# @pytest.fixture(scope="module")
# def gamma_distri(databook):
#     pass


# @pytest.fixture(scope="module")
# def loglogistic_distri(databook):
#     pass


@pytest.mark.parametrize(
    "model, params",
    [
        ("exponential_distri", 0.00795203),
        # ("weibull_distri", [3.46597395, 0.01227849]),
        # ("gompertz_distri", [0.00865741, 0.06062632]),
        # ("gamma_distri", [5.3571091, 0.06622822]),
        # ("loglogistic_distri", [3.92614064, 0.0133325]),
    ],
)
def test_sf(model, params, request):
    model = request.getfixturevalue(model)
    print(model.params)
    assert model.sf(
        model.median(params=params), params=params
    ) == pytest.approx(0.5, rel=1e-3)


@pytest.mark.parametrize(
    "model, params",
    [
        ("exponential_distri", 0.00795203),
        # ("weibull_distri", [3.46597395, 0.01227849]),
        # ("gompertz_distri", [0.00865741, 0.06062632]),
        # ("gamma_distri", [5.3571091, 0.06622822]),
        # ("loglogistic_distri", [3.92614064, 0.0133325]),
    ],
)
def test_rvs(model, params, request):
    model = request.getfixturevalue(model)
    size = 10
    assert model.rvs(size=size, params=params).shape == (size,)


# /!\ depends upon LS_INTEGRATE

# @pytest.mark.parametrize(
#     "model, params",
#     [
#         ("exponential_distri", 0.00795203),
#         # ("weibull_distri", [3.46597395, 0.01227849]),
#         # ("gompertz_distri", [0.00865741, 0.06062632]),
#         # ("gamma_distri", [5.3571091, 0.06622822]),
#         # ("loglogistic_distri", [3.92614064, 0.0133325]),
#     ],
# )
# def test_mean(model, params, request):
#     model = request.getfixturevalue(model)
#     assert super(type(model), model).mean(params=params) == pytest.approx(
#         model.mean(params=params), rel=1e-3
#     )


@pytest.mark.parametrize(
    "model, params",
    [
        ("exponential_distri", 0.00795203),
        # ("weibull_distri", [3.46597395, 0.01227849]),
        # ("gompertz_distri", [0.00865741, 0.06062632]),
        # ("gamma_distri", [5.3571091, 0.06622822]),
        # ("loglogistic_distri", [3.92614064, 0.0133325]),
    ],
)
def test_mrl(model, params, request):
    model = request.getfixturevalue(model)
    t = np.arange(10)
    assert model.mrl(t, params=params).shape == (t.size,)


@pytest.mark.parametrize(
    "model, params",
    [
        ("exponential_distri", 0.00795203),
        # ("weibull_distri", [3.46597395, 0.01227849]),
        # ("gompertz_distri", [0.00865741, 0.06062632]),
        # ("gamma_distri", [5.3571091, 0.06622822]),
        # ("loglogistic_distri", [3.92614064, 0.0133325]),
    ],
)
def test_fit(model, params, request):
    model = request.getfixturevalue(model)
    model.fit()
    assert model.fitting_params == pytest.approx(params, rel=1e-3)


# def test_minimum_distribution(model, data):
#     params = model.params.copy()
#     n = np.ones((data.size, 1))
#     model = MinimumDistribution(model).fit(*data.astuple(), args=(n,))
#     assert model.params == pytest.approx(params, rel=1e-3)
