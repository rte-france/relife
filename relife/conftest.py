# pyright: basic
import numpy as np
import pytest
from numpy.typing import NDArray

from relife.data import load_insulator_string, load_power_transformer
from relife.lifetime_model import (
    AgeReplacementModel,
    Exponential,
    Gamma,
    Gompertz,
    LogLogistic,
    ParametricAcceleratedFailureTime,
    ParametricProportionalHazard,
    Weibull,
)

#######################################################################################
# DATA FIXTURES
#######################################################################################


@pytest.fixture
def power_transformer_data():
    return load_power_transformer()


@pytest.fixture
def insulator_string_data():
    return load_insulator_string()


#######################################################################################
# LIFETIME MODEL FIXTURES
#######################################################################################


def exponential():
    return Exponential(0.00795203)


def weibull():
    return Weibull(3.46597395, 0.01227849)


def gompertz():
    return Gompertz(0.00865741, 0.06062632)


def gamma():
    return Gamma(5.3571091, 0.06622822)


def loglogistic():
    return LogLogistic(3.92614064, 0.0133325)


COEFFICIENTS = (np.log(2), np.log(2))


def pph_exponential():
    return ParametricProportionalHazard(exponential(), coefficients=COEFFICIENTS)


def pph_weibull():
    return ParametricProportionalHazard(weibull(), coefficients=COEFFICIENTS)


def pph_gompertz():
    return ParametricProportionalHazard(gompertz(), coefficients=COEFFICIENTS)


def pph_gamma():
    return ParametricProportionalHazard(gamma(), coefficients=COEFFICIENTS)


def pph_loglogistic():
    return ParametricProportionalHazard(loglogistic(), coefficients=COEFFICIENTS)


def aft_exponential():
    return ParametricAcceleratedFailureTime(exponential(), coefficients=COEFFICIENTS)


def aft_weibull():
    return ParametricAcceleratedFailureTime(weibull(), coefficients=COEFFICIENTS)


def aft_gompertz():
    return ParametricAcceleratedFailureTime(gompertz(), coefficients=COEFFICIENTS)


def aft_gamma():
    return ParametricAcceleratedFailureTime(gamma(), coefficients=COEFFICIENTS)


def aft_loglogistic():
    return ParametricAcceleratedFailureTime(loglogistic(), coefficients=COEFFICIENTS)


@pytest.fixture(params=[exponential(), weibull(), gompertz(), gamma(), loglogistic()])
def distribution(request):
    yield request.param


@pytest.fixture(
    params=[
        pph_exponential(),
        pph_weibull(),
        pph_gompertz(),
        pph_gamma(),
        pph_loglogistic(),
        aft_exponential(),
        aft_weibull(),
        aft_gompertz(),
        aft_gamma(),
        aft_loglogistic(),
    ]
)
def regression(request):
    yield request.param


#######################################################################################
# LIFETIME LIKELIHOOD FIXTURES
#######################################################################################


@pytest.fixture
def distribution_likelihood(distribution, power_transformer_data):
    return distribution.init_likelihood(
        power_transformer_data["time"],
        event=power_transformer_data["event"],
        entry=power_transformer_data["entry"],
    )


@pytest.fixture
def regression_likelihood(regression, insulator_string_data):
    covar = np.column_stack(
        (
            insulator_string_data["pHCl"],
            insulator_string_data["pH2SO4"],
        )
    )
    return regression.init_likelihood(
        insulator_string_data["time"],
        covar,
        event=insulator_string_data["event"],
        entry=insulator_string_data["entry"],
    )


#######################################################################################
# FROZEN LIFETIME FIXTURES
#######################################################################################

NB_ASSETS = 3


@pytest.fixture
def frozen_regression(regression):
    covar = np.linspace(0.0, 0.5, num=NB_ASSETS * regression.nb_coef).reshape(
        NB_ASSETS, regression.nb_coef
    )
    return regression.freeze(covar)


@pytest.fixture
def frozen_ar_distribution(distribution):
    ar = distribution.isf(0.75)
    return AgeReplacementModel(distribution).freeze(ar)


@pytest.fixture
def frozen_ar_regression(regression):
    covar = np.linspace(0.0, 0.5, num=NB_ASSETS * regression.nb_coef).reshape(
        NB_ASSETS, regression.nb_coef
    )
    ar = regression.isf(0.75, covar)
    return AgeReplacementModel(regression).freeze(ar, covar)


#######################################################################################
# LIFETIME MODEL VARIABLES FIXTURES
#######################################################################################

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
        np.ones((len(COEFFICIENTS),), dtype=np.float64),
        np.ones((1, len(COEFFICIENTS)), dtype=np.float64),
        np.ones((M, len(COEFFICIENTS)), dtype=np.float64),
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


#######################################################################################
# ECONOMIC FIXTURES
#######################################################################################


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


#######################################################################################
# FACTORY FIXTURE TO CONTROL IO SHAPES
#######################################################################################


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
