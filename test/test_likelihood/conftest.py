import numpy as np
import pytest

from relife.data import load_power_transformer, load_insulator_string, LifetimeData
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


@pytest.fixture
def lifetime_data_1d():
    time = np.array([10, 11, 9, 10, 12, 13, 11], dtype=np.float64)
    event = np.array([1, 0, 1, 0, 0, 0, 1], dtype=np.bool_)
    entry = np.array([0, 0, 3, 5, 3, 1, 9], dtype=np.float64)
    return LifetimeData(time, event=event, entry=entry)

@pytest.fixture
def lifetime_data_2d():
    time = np.array(
        [[1, 2], [0, 4], [5, 5], [7, np.inf], [10, 10], [2, 10], [10, 11]],
        dtype=np.float64
    )
    entry = np.array([0, 0, 3, 5, 3, 1, 9], dtype=np.float64)
    departure = np.array([4, np.inf, 7, 10, np.inf, 12, np.inf], dtype=np.float64)
    return LifetimeData(time, entry=entry, departure=departure)


@pytest.fixture
def power_transformer_data():
    return load_power_transformer()

@pytest.fixture
def insulator_string_data():
    return load_insulator_string()