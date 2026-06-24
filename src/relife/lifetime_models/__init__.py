"""
``relife.lifetime_model``
=========================

The ReLife lifetime_model module exposes various stochastic models to modelize
lifetime data. Internal operations are computed using NumPy and Scipy.

- NumPy: https://github.com/numpy/numpy
- Scipy: https://github.com/scipy/scipy

Objects present in relife.lifetime_model are listed below.

Lifetime distributions
----------------------

    Exponential
    Weibull
    Gompertz
    Gamma
    LogLogistic
    MinimumDistribution
    EquilibriumDistribution


Lifetime regressions
--------------------

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

Likelihoods
-----------

    LifetimeLikelihood
    CoxPartialLifetimeLikelihood
    BreslowPartialLifetimeLikelihood
    EfronPartialLifetimeLikelihood

"""

from ._base import ParametricLifetimeModel
from ._distributions import (
    EquilibriumDistribution,
    Exponential,
    Gamma,
    Gompertz,
    LifetimeDistribution,
    LogLogistic,
    Weibull,
)
from ._non_parametric_models import ECDF, KaplanMeier, NelsonAalen
from ._parametric_regressions import (
    ParametricAcceleratedFailureTime,
    ParametricLifetimeRegression,
    ParametricProportionalHazard,
)
from ._semi_parametric_regressions import SemiParametricProportionalHazard

__all__: list[str] = []
__all__ += ["ParametricLifetimeModel"]
__all__ += [
    "EquilibriumDistribution",
    "Exponential",
    "Gamma",
    "Gompertz",
    "LifetimeDistribution",
    "LogLogistic",
    "Weibull",
]
__all__ += [
    "ParametricAcceleratedFailureTime",
    "ParametricLifetimeRegression",
    "ParametricProportionalHazard",
]
__all__ += ["ECDF", "KaplanMeier", "NelsonAalen"]
__all__ += ["SemiParametricProportionalHazard"]
