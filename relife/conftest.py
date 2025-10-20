import numpy as np
import pytest
from numpy.typing import NDArray

from relife.data import load_insulator_string, load_power_transformer
from relife.lifetime_model import (
    AcceleratedFailureTime,
    AgeReplacementModel,
    Exponential,
    Gamma,
    Gompertz,
    LogLogistic,
    ProportionalHazard,
    Weibull,
)

DISTRIBUTION_INSTANCES = [
    Exponential(0.00795203),
    Weibull(3.46597395, 0.01227849),
    Gompertz(0.00865741, 0.06062632),
    Gamma(5.3571091, 0.06622822),
    LogLogistic(3.92614064, 0.0133325),
]

NB_COEF = 2

REGRESSION_INSTANCES = [
    ProportionalHazard(Exponential(0.00795203), coefficients=(np.log(2), np.log(2))),
    ProportionalHazard(Weibull(3.46597395, 0.01227849), coefficients=(np.log(2), np.log(2))),
    ProportionalHazard(Gompertz(0.00865741, 0.06062632), coefficients=(np.log(2), np.log(2))),
    ProportionalHazard(Gamma(5.3571091, 0.06622822), coefficients=(np.log(2), np.log(2))),
    ProportionalHazard(LogLogistic(3.92614064, 0.0133325), coefficients=(np.log(2), np.log(2))),
    AcceleratedFailureTime(Exponential(0.00795203), coefficients=(np.log(2), np.log(2))),
    AcceleratedFailureTime(Weibull(3.46597395, 0.01227849), coefficients=(np.log(2), np.log(2))),
    AcceleratedFailureTime(Gompertz(0.00865741, 0.06062632), coefficients=(np.log(2), np.log(2))),
    AcceleratedFailureTime(Gamma(5.3571091, 0.06622822), coefficients=(np.log(2), np.log(2))),
    AcceleratedFailureTime(LogLogistic(3.92614064, 0.0133325), coefficients=(np.log(2), np.log(2))),
]

########################################################################################################################
# DATA FIXTURES
########################################################################################################################


@pytest.fixture
def power_transformer_data():
    return load_power_transformer()


@pytest.fixture
def insulator_string_data():
    return load_insulator_string()

########################################################################################################################
# LIFETIME MODEL FIXTURES
########################################################################################################################


@pytest.fixture(
    params=DISTRIBUTION_INSTANCES,
    ids=["Exponential", "Weibull", "Gompertz", "Gamma", "LogLogistic"],
)
def distribution(request):
    return request.param


@pytest.fixture(
    params=REGRESSION_INSTANCES,
    ids=lambda reg: f"{reg.__class__.__name__}({reg.baseline.__class__.__name__})",
)
def regression(request):
    return request.param


########################################################################################################################
# FROZEN LIFETIME FIXTURES
########################################################################################################################


@pytest.fixture(
    params=REGRESSION_INSTANCES,
    ids=lambda reg: f"Frozen{reg.__class__.__name__}({reg.baseline.__class__.__name__})",
)
def frozen_regression(request):
    covar = np.arange(0.0, 0.6, 0.1).reshape(3, 2)
    return request.param.freeze(covar)


@pytest.fixture(
    params=DISTRIBUTION_INSTANCES,
    ids=lambda distri: f"FrozenAgeReplacementModel({distri.__class__.__name__})",
)
def frozen_ar_distribution(request):
    return AgeReplacementModel(request.param).freeze(request.param.isf(0.75))


@pytest.fixture(
    params=REGRESSION_INSTANCES,
    ids=lambda reg: f"FrozenAgeReplacementModel({reg.__class__.__name__}({reg.baseline.__class__.__name__}))",
)
def frozen_ar_regression(request):
    covar = np.arange(0.0, 0.6, 0.1).reshape(3, 2)
    return AgeReplacementModel(request.param).freeze(request.param.isf(0.75, covar), covar)


########################################################################################################################
# LIFETIME MODEL VARIABLES FIXTURES
########################################################################################################################

M = 3  # nb assets
N = 10  # nb points


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
        np.float64(0.5),
        np.ones((1,), dtype=np.float64) * 0.5,
        np.ones((N,), dtype=np.float64) * 0.5,
        np.ones((1, 1), dtype=np.float64) * 0.5,
        np.ones((M, 1), dtype=np.float64) * 0.5,
        np.ones((1, N), dtype=np.float64) * 0.5,
        np.ones((M, N), dtype=np.float64) * 0.5,
    ],
    ids=lambda probability: f"probability:{probability.shape}",
)
def probability(request):
    return request.param


@pytest.fixture(
    params=[
        np.ones((NB_COEF,), dtype=np.float64),
        np.ones((1, NB_COEF), dtype=np.float64),
        np.ones((M, NB_COEF), dtype=np.float64),
    ],
    ids=lambda covar: f"covar:{covar.shape}",
)
def covar(request):
    return request.param


@pytest.fixture(
    params=[
        2.0 * np.ones((), dtype=np.float64),
        2.0 * np.ones((1,), dtype=np.float64),
        2.0 * np.ones((N,), dtype=np.float64),
        2.0 * np.ones((1, N), dtype=np.float64),
        2.0 * np.ones((M, 1), dtype=np.float64),
        2.0 * np.ones((M, N), dtype=np.float64),
    ],
    ids=lambda a: f"a:{a.shape}",
)
def integration_bound_a(request):
    return request.param


@pytest.fixture(
    params=[
        8.0 * np.ones((), dtype=np.float64),
        8.0 * np.ones((1,), dtype=np.float64),
        8.0 * np.ones((N,), dtype=np.float64),
        8.0 * np.ones((1, N), dtype=np.float64),
        8.0 * np.ones((M, 1), dtype=np.float64),
        8.0 * np.ones((M, N), dtype=np.float64),
    ],
    ids=lambda b: f"b:{b.shape}",
)
def integration_bound_b(request):
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
        np.ones((), dtype=np.float64),
        np.ones((1,), dtype=np.float64),
        np.ones((M,), dtype=np.float64),
        np.ones((M, 1), dtype=np.float64),
    ],
    ids=lambda a0: f"a0:{a0.shape}",
)
def a0(request):
    return request.param


@pytest.fixture(
    params=[1, N],
    ids=lambda size: f"size:{size}",
)
def rvs_size(request):
    return request.param


@pytest.fixture(
    params=[None, M],
    ids=lambda nb_assets: f"nb_assets:{nb_assets}",
)
def rvs_nb_assets(request):
    return request.param


########################################################################################################################
# ECONOMIC FIXTURES
########################################################################################################################


@pytest.fixture(
    params=[
        np.ones((), dtype=np.float64),
        # np.ones((1,), dtype=np.float64),
        np.ones((M,), dtype=np.float64),
        # np.ones((M, 1), dtype=np.float64),
    ],
    ids=lambda cp: f"cp:{cp.shape}",
)
def cp(request):
    return request.param


# M = 3
CF_RANGE = [5, 10, 20, 100, 1000]


@pytest.fixture(
    params=[
        np.array(CF_RANGE[0], dtype=np.float64),  # ()
        # np.array([CF_RANGE[0]], dtype=np.float64), # (1,)
        np.array(CF_RANGE[:M], dtype=np.float64),  # (M,)
        # np.array(CF_RANGE[:M], dtype=np.float64).reshape(-1, 1), # (M, 1)
    ],
    ids=lambda cf: f"cf:{cf.shape}",
)
def cf(request):
    return request.param


@pytest.fixture(params=[0.0, 0.04], ids=lambda rate: f"discounting_rate:{rate}")
def discounting_rate(request):
    return request.param


########################################################################################################################
# FACTORY FIXTURE TO CONTROL IO SHAPES
########################################################################################################################


@pytest.fixture
def expected_out_shape():
    def _expected_out_shape(**kwargs: NDArray[np.float64]) -> tuple[int, ...]:
        def shape_contrib(**kwargs: NDArray[np.float64]):
            yield ()  # yield at least (), in case kwargs is empty
            for k, v in kwargs.items():
                match k:
                    case "covar" if v.ndim == 2:
                        yield v.shape[0], 1
                    case "covar" if v.ndim < 2:
                        yield ()
                    case "cf" | "cp" | "ar" | "a0" if v.ndim == 2 or v.ndim == 0:
                        yield v.shape
                    case "cf" | "cp" | "ar" | "a0" if v.ndim == 1:
                        yield v.size, 1
                    case "size":
                        if isinstance(v, int):
                            if v == 1:
                                yield ()
                            yield (v,)
                        yield v  # it is tuple
                    case _:
                        yield v.shape

        return np.broadcast_shapes(*tuple(shape_contrib(**kwargs)))

    return _expected_out_shape
