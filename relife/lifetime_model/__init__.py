"""
``relife.lifetime_model``
=========================

The ReLife lifetime_model module exposes various stochastic models to modelize
lifetime data. Internal operations are computed using NumPy and Scipy.

- NumPy: https://github.com/numpy/numpy
- Scipy: https://github.com/scipy/scipy

Objects present in relife.lifetime_model are listed below.

Lifetime distribution
---------------------

    Exponential
    Weibull
    Gompertz
    Gamma
    LogLogistic
    MinimumDistribution
    EquilibriumDistribution


Lifetime regression
-------------------

    ProportionalHazard
    AcceleratedFailureTime

Nonparametric models
--------------------

    KaplanMeier
    ECDF
    NelsonAalen

Conditional models
------------------

    AgeReplacementModel
    LeftTruncatedModel

"""

from ._conditional_model import AgeReplacementModel, LeftTruncatedModel
from ._distribution import (
    EquilibriumDistribution,
    Exponential,
    Gamma,
    Gompertz,
    LogLogistic,
    MinimumDistribution,
    Weibull,
)
from ._parametric import ParametricProportionalHazard, ParametricAcceleratedFailureTime
from ._semi_parametric import SemiParametricProportionalHazard
from ._non_parametric import ECDF, KaplanMeier, NelsonAalen

__all__ = [
    "Exponential",
    "Weibull",
    "Gompertz",
    "Gamma",
    "LogLogistic",
    "MinimumDistribution",
    "EquilibriumDistribution",
    "ParametricProportionalHazard",
    "ParametricAcceleratedFailureTime",
    "SemiParametricProportionalHazard",
    "KaplanMeier",
    "ECDF",
    "NelsonAalen",
    "AgeReplacementModel",
    "LeftTruncatedModel",
]
