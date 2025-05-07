import numpy as np
import pytest
from numpy.typing import NDArray

from relife.data import load_power_transformer, load_insulator_string
from relife.lifetime_model import (
    Exponential,
    Weibull,
    LogLogistic,
    Gamma,
    Gompertz,
    ProportionalHazard,
    AcceleratedFailureTime,
)

NB_COEF = 3

@pytest.fixture(
    params=[
        Exponential(0.00795203),
        Weibull(3.46597395, 0.01227849),
        Gompertz(0.00865741, 0.06062632),
        Gamma(5.3571091, 0.06622822),
        LogLogistic(3.92614064, 0.0133325),
    ],
    ids=["Exponential", "Weibull", "Gompertz", "Gamma", "LogLogistic"]
)
def distribution(request):
    return request.param


@pytest.fixture(
    params=[
        ProportionalHazard(Exponential(0.00795203), coefficients=(0.1,) * NB_COEF),
        ProportionalHazard(Weibull(3.46597395, 0.01227849), coefficients=(0.1,) * NB_COEF),
        ProportionalHazard(Gompertz(0.00865741, 0.06062632), coefficients=(0.1,) * NB_COEF),
        ProportionalHazard(Gamma(5.3571091, 0.06622822), coefficients=(0.1,) * NB_COEF),
        ProportionalHazard(LogLogistic(3.92614064, 0.0133325), coefficients=(0.1,) * NB_COEF),
        AcceleratedFailureTime(Exponential(0.00795203), coefficients=(0.1,) * NB_COEF),
        AcceleratedFailureTime(Weibull(3.46597395, 0.01227849), coefficients=(0.1,) * NB_COEF),
        AcceleratedFailureTime(Gompertz(0.00865741, 0.06062632), coefficients=(0.1,) * NB_COEF),
        AcceleratedFailureTime(Gamma(5.3571091, 0.06622822), coefficients=(0.1,) * NB_COEF),
        AcceleratedFailureTime(LogLogistic(3.92614064, 0.0133325), coefficients=(0.1,) * NB_COEF),
    ],
    ids=lambda reg : f"{reg.__class__.__name__}({reg.baseline.__class__.__name__})"
)
def regression(request):
    return request.param


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
    ids=lambda time : f"time{time.shape}"
)
def time(request):
    return request.param

@pytest.fixture(
    params=[
        np.float64(0.5),
        np.ones((1,), dtype=np.float64) * 0.5,
        np.ones((N,), dtype=np.float64) * 0.5,
        np.ones((1, 1), dtype=np.float64) * 0.5,
        np.ones((M, 1), dtype=np.float64) * 0.5,
        np.ones((1, N), dtype=np.float64) * 0.5,
        np.ones((M, N), dtype=np.float64) * 0.5,
    ],
    ids=lambda probability : f"probability{probability.shape}"
)
def probability(request):
    return request.param

@pytest.fixture(
    params=[
        np.ones((NB_COEF,), dtype=np.float64),
        np.ones((1, NB_COEF), dtype=np.float64),
        np.ones((M, NB_COEF), dtype=np.float64)
    ],
    ids=lambda covar : f"covar{covar.shape}"
)
def covar(request):
    return request.param

@pytest.fixture(
    params = [
        2.0 * np.ones((), dtype=np.float64),
        2.0 * np.ones((1,), dtype=np.float64),
        2.0 * np.ones((N,), dtype=np.float64),
        2.0 * np.ones((1,N), dtype=np.float64),
        2.0 * np.ones((M,1), dtype=np.float64),
        2.0 * np.ones((M,N), dtype=np.float64),
    ],
    ids = lambda a : f"a{a.shape}"
)
def integration_bound_a(request):
    return request.param

@pytest.fixture(
    params = [
        8.0 * np.ones((), dtype=np.float64),
        8.0 * np.ones((1,), dtype=np.float64),
        8.0 * np.ones((N,), dtype=np.float64),
        8.0 * np.ones((1,N), dtype=np.float64),
        8.0 * np.ones((M,1), dtype=np.float64),
        8.0 * np.ones((M,N), dtype=np.float64),
    ],
    ids = lambda b: f"b{b.shape}"
)
def integration_bound_b(request):
    return request.param

@pytest.fixture(
    params = [
        np.ones((), dtype=np.float64),
        np.ones((1,), dtype=np.float64),
        np.ones((M,), dtype=np.float64),
        np.ones((M,1), dtype=np.float64),
    ],
    ids = lambda ar : f"ar{ar.shape}"
)
def ar(request):
    return request.param

@pytest.fixture(
    params = [
        np.ones((), dtype=np.float64),
        np.ones((1,), dtype=np.float64),
        np.ones((M,), dtype=np.float64),
        np.ones((M,1), dtype=np.float64),
    ],
    ids = lambda a0 : f"a0{a0.shape}"
)
def a0(request):
    return request.param


@pytest.fixture
def expected_out_shape():
    def _expected_out_shape(**kwargs: NDArray[np.float64]):
        def shape_contrib(**kwargs: NDArray[np.float64]):
            yield () # yield at least (), in case kwargs is empty
            for k, v in kwargs.items():
                match k:
                    case "covar" if v.ndim == 2:
                        yield v.shape[0], 1
                    case "covar" if v.ndim < 2:
                        yield ()
                    case "ar"|"a0" if v.ndim ==2:
                        yield v.shape
                    case "ar"|"a0" if v.ndim < 2:
                        yield v.size, 1
                    case _:
                        yield v.shape
        return np.broadcast_shapes(*tuple(shape_contrib(**kwargs)))
    return _expected_out_shape


@pytest.fixture
def power_transformer_data():
    return load_power_transformer()

@pytest.fixture
def insulator_string_data():
    return load_insulator_string()