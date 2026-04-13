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

    ParametricProportionalHazard
    ParametricAcceleratedFailureTime

Semiparametric lifetime regression
----------------------------------

    SemiParametricProportionalHazard


Nonparametric models
--------------------

    KaplanMeier
    ECDF
    NelsonAalen

Conditional models
------------------

    AgeReplacementModel
    LeftTruncatedModel


Likelihoods
-----------

    LifetimeLikelihood
    CoxPartialLifetimeLikelihood
    BreslowPartialLifetimeLikelihood
    EfronPartialLifetimeLikelihood

"""

from ._base import AnyParametricLifetimeModel, LifetimeLikelihood
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
from ._non_parametric import ECDF, KaplanMeier, NelsonAalen
from ._regression import ParametricAcceleratedFailureTime, ParametricProportionalHazard
from ._semi_parametric import (
    BreslowPartialLifetimeLikelihood,
    CoxPartialLifetimeLikelihood,
    EfronPartialLifetimeLikelihood,
    SemiParametricProportionalHazard,
)

__all__ = [
    "LifetimeLikelihood",
    "AnyParametricLifetimeModel",
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
    "CoxPartialLifetimeLikelihood",
    "EfronPartialLifetimeLikelihood",
    "BreslowPartialLifetimeLikelihood",
]
