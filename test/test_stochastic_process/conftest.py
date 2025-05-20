import pytest
import numpy as np

from relife import freeze
from relife.lifetime_model import (
    Exponential,
    Weibull,
    Gompertz,
    Gamma,
    LogLogistic,
    ProportionalHazard,
    AcceleratedFailureTime,
    AgeReplacementModel,
)


@pytest.fixture(
    params=[
        Weibull(3.46597395, 0.01227849),
        Gompertz(0.00865741, 0.06062632),
        Gamma(5.3571091, 0.06622822),
        LogLogistic(3.92614064, 0.0133325),
    ],
    ids=lambda distri: f"{distri.__class__.__name__}",
)
def distribution(request):
    return request.param


@pytest.fixture(
    params=[
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
    ],
    ids=lambda reg: f"Frozen{reg.__class__.__name__}({reg.baseline.__class__.__name__})",
)
def frozen_regression(request):
    covar = np.arange(0.0, 0.6, 0.1).reshape(3, 2)
    return freeze(request.param, covar=covar)


@pytest.fixture(
    params=[
        Weibull(3.46597395, 0.01227849),
        Gompertz(0.00865741, 0.06062632),
        Gamma(5.3571091, 0.06622822),
        LogLogistic(3.92614064, 0.0133325),
    ],
    ids=lambda distri: f"FrozenAgeReplacementModel({distri.__class__.__name__})",
)
def frozen_ar_distribution(request):
    ar = request.param.isf(0.75)
    return freeze(AgeReplacementModel(request.param), ar=ar)


@pytest.fixture(
    params=[
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
    ],
    ids=lambda reg: f"FrozenAgeReplacementModel({reg.__class__.__name__}({reg.baseline.__class__.__name__}))",
)
def frozen_ar_regression(request):
    covar = np.arange(0.0, 0.6, 0.1).reshape(3, 2)
    ar = request.param.isf(0.75, covar)
    return freeze(AgeReplacementModel(request.param), ar=ar, covar=covar)
